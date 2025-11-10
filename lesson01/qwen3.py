import json
import os
from glob import glob
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import Qwen2Tokenizer

"""
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
"""


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position: int = 65536,
        base: float = 10000,
    ):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        m = torch.arange(max_position)
        m_theta = torch.outer(m, theta)
        cos = m_theta.cos()  # [max_position, head_dim//2]
        sin = m_theta.sin()  # [max_position, head_dim//2]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, query, key):
        # input: [batch_size, num_heads, seq_len, head_dim]
        return self.apply_rotary_emb(query), self.apply_rotary_emb(key)

    def apply_rotary_emb(self, x):
        # input: [batch_size, num_heads, seq_len, head_dim]
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len]  # [seq_len, head_dim//2]
        sin = self.sin[:seq_len]  # [seq_len, head_dim//2]
        x1, x2 = x.chunk(2, dim=-1)  # [batch_size, num_heads, seq_len, head_dim//2]
        y1 = x1 * cos - x2 * sin  # [batch_size, num_heads, seq_len, head_dim//2]
        y2 = x2 * cos + x1 * sin  # [batch_size, num_heads, seq_len, head_dim//2]
        return torch.cat([y1, y2], dim=-1)  # [batch_size, num_heads, seq_len, head_dim]


def make_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32):
    device = attention_mask.device
    # input: [batch_size, seq_len]
    seq_len = attention_mask.shape[1]
    # [seq_len, seq_len]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
    # [batch_size, seq_len, seq_len]
    pad_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
    # output: [batch_size, seq_len, seq_len]
    return (1 - causal_mask * pad_mask) * torch.finfo(dtype).min


def repeat_kv(x: torch.Tensor, n_rep: int):
    # input: [batch_size, num_heads, seq_len, head_dim]
    # 1. unsqueeze -> [batch_size, num_heads, 1, seq_len, head_dim]
    # 2. expand -> [batch_size, num_heads, n_rep, seq_len, head_dim]
    shape = x.shape
    x = x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1)
    # output: [batch_size, num_heads * n_rep, seq_len, head_dim]
    return x.reshape(shape[0], shape[1] * n_rep, shape[2], shape[3])


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        # [hidden_size]
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        # input: [batch_size, seq_len, hidden_size]
        # [batch_size, seq_len, 1]
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # output: [batch_size, seq_len, hidden_size]
        return hidden_states * torch.rsqrt(variance + self.eps) * self.weight


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        rotary_emb: nn.Module = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.rotary_emb = rotary_emb if rotary_emb else RotaryEmbedding(head_dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # input: [batch_size, seq_len, hidden_size]
        input_shape = hidden_states.shape[:-1]
        # 切成多头后的形状 [batch_size, seq_len, ?, head_dim]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 1. q_proj -> [batch_size, seq_len, num_heads * head_dim]
        # 2. view -> [batch_size, seq_len, num_heads, head_dim]
        # 3. norm -> keep
        # 4. transpose -> [batch_size, num_heads, seq_len, head_dim]
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # 1. k_proj -> [batch_size, seq_len, num_kv_heads * head_dim]
        # 2. view -> [batch_size, seq_len, num_kv_heads, head_dim]
        # 3. norm -> keep
        # 4. transpose -> [batch_size, num_kv_heads, seq_len, head_dim]
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # apply rotary -> keep
        q, k = self.rotary_emb(q, k)
        # repeat kv -> [batch_size, num_heads, seq_len, head_dim]
        kk = repeat_kv(k, self.num_kv_groups)
        vv = repeat_kv(v, self.num_kv_groups)
        # S = QK^T / sqrt(d) -> [batch_size, num_heads, seq_len, seq_len]
        s = torch.matmul(q, kk.transpose(-2, -1)) * self.scaling
        # causal mask : [batch_size, seq_len, seq_len]
        causal_mask = make_causal_mask(attention_mask)
        # apply causal mask -> [batch_size, num_heads, seq_len, seq_len]
        s = s + causal_mask[:, None, :, :]
        # apply softmax -> [batch_size, num_heads, seq_len, seq_len]
        s = nn.functional.softmax(s, dim=-1)
        # O = SV -> [batch_size, num_heads, seq_len, head_dim]
        o = torch.matmul(s, vv)
        # 1. transpose -> [batch_size, seq_len, num_heads, head_dim]
        # 2. reshape -> [batch_size, seq_len, num_heads * head_dim]
        o = o.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        # o_proj -> [batch_size, seq_len, hidden_size]
        o = self.o_proj(o)
        return o


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        # input: [batch_size, seq_len, hidden_size]
        # 1. gate_proj -> [batch_size, seq_len, intermediate_size]
        # 2. act_fn -> keep
        act_gate = self.act_fn(self.gate_proj(hidden_states))
        # up_proj -> [batch_size, seq_len, intermediate_size]
        up = self.up_proj(hidden_states)
        # down_proj -> [batch_size, seq_len, hidden_size]
        return self.down_proj(act_gate * up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
        )
        self.self_attn = Qwen3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            self.rotary_emb,
        )
        self.mlp = Qwen3MLP(
            config.hidden_size,
            config.intermediate_size,
            nn.functional.silu,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states[:, -1:, :])


def load_model_config(dir_path: str):
    path = os.path.join(dir_path, "config.json")
    config_obj = json.load(open(path))
    path = os.path.join(dir_path, "generation_config.json")
    config_obj.update(json.load(open(path)))
    return SimpleNamespace(**config_obj)


def load_model(model: nn.Module, dir_path: str):
    path = os.path.join(dir_path, "*.safetensors")
    for file in glob(path):
        print(f"Reading weights from {file}")
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param = model.get_parameter(weight_name)
                param.data.copy_(f.get_tensor(weight_name))
    print("Model weights loaded")


model_weight_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

model_config = load_model_config(model_weight_path)
model = Qwen3ForCausalLM(model_config)

load_model(model, model_weight_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model is on device: {device}")

tokenizer = Qwen2Tokenizer.from_pretrained(model_weight_path)
tokenizer.padding_side = "left"  # 推理需要左侧填充

prompts = [
    "Give me a short introduction to large language model.",
    "1 + 1 = ?",
]
chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for messages in chats
]
model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)


def get_next_inputs(
    last_inputs: dict[str, torch.Tensor],
    sample_tokens: torch.Tensor,
    eos_token_ids: list[int],
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:

    batch_size = sample_tokens.size(0)
    next_inputs = {}
    finished_inputs = []

    next_inputs["input_ids"] = torch.cat(
        [last_inputs["input_ids"], sample_tokens], dim=-1
    )
    next_inputs["attention_mask"] = torch.cat(
        [last_inputs["attention_mask"], torch.ones_like(sample_tokens)], dim=-1
    )

    for i in range(batch_size):
        if sample_tokens[i].item() in eos_token_ids:
            finished_inputs.append(next_inputs["input_ids"][i])

    if len(finished_inputs) == batch_size:
        next_inputs["input_ids"] = torch.empty(0)
        next_inputs["attention_mask"] = torch.empty(0)
    elif len(finished_inputs) > 0:
        next_inputs["input_ids"] = torch.cat(
            [
                next_inputs["input_ids"][i : i + 1]
                for i in range(batch_size)
                if sample_tokens[i].item() not in eos_token_ids
            ],
            dim=0,
        )
        next_inputs["attention_mask"] = torch.cat(
            [
                next_inputs["attention_mask"][i : i + 1]
                for i in range(batch_size)
                if sample_tokens[i].item() not in eos_token_ids
            ],
            dim=0,
        )

    return next_inputs, finished_inputs


def step(model: nn.Module, model_inputs: dict[str, torch.Tensor]):
    # output: [batch_size, 1]
    return model.compute_logits(model(**model_inputs)).argmax(dim=-1)


@torch.inference_mode
def generate(
    model: nn.Module,
    model_config,
    model_inputs: dict[str, torch.Tensor],
    max_length: int = 100,
):
    finished = []
    next_inputs = model_inputs
    while next_inputs["input_ids"].shape[0]:
        sample_tokens = step(model, next_inputs)
        next_inputs, next_finished = get_next_inputs(
            next_inputs, sample_tokens, model_config.eos_token_id
        )
        finished.extend(next_finished)
        if next_inputs["input_ids"].shape[-1] >= max_length:
            finished.extend(next_inputs["input_ids"])
            break
    return finished


outputs = generate(model, model_config, model_inputs)

for i in range(len(outputs)):
    print("-" * 40)
    print(tokenizer.decode(outputs[i]))
