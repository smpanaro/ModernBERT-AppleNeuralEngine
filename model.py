import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional

"""
nanoGPT-inspired, pared down version of the HF ModernGPT implementation,
optimized for Apple Neural Engine a la ml-ane-transformers + ml-stable-diffusion.
"""

@dataclass
class Config:
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int
    local_rope_theta: int = 10000
    global_rope_theta: int = 160000
    global_attn_every_n_layers: int = 3
    local_attention_window_size: int = 128 # HF "local_attention"
    vocab_size: int = 50368
    pad_token_id: int = 50283
    norm_eps: float = 1e-5
    norm_bias: bool = False
    match_hf: bool = False # Adjust to match HF more closely.

class Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.norm(self.embeddings(input_ids))

class Attention(nn.Module):
    def __init__(self, config: Config, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.qkv = nn.Conv2d(config.hidden_size, config.hidden_size * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1, bias=False)
        self.dim_head = config.hidden_size // config.num_heads

        self.mask_min_value = -1e4 # torch.finfo(attention_mask.dtype).min is too small for ANE / float16.

        self.use_global_attention = self.layer_index % self.config.global_attn_every_n_layers == 0
        rope_theta = config.global_rope_theta if self.use_global_attention else config.local_rope_theta
        self.rotary_emb = ModernBertRotaryEmbedding(dim=self.dim_head, base=rope_theta)

    def forward(self, x, position_ids, attention_mask):
        """
        x: (bs, hidden_size, 1, seq_length)
        position_ids: (bs, seq_length)
        attention_mask: (bs, 1, seq_length, seq_length)
        """
        b, c, _, s = x.shape

        if self.config.match_hf:
            # Slight numerical accuracy drift between this Linear and Conv2d.
            qkv = F.linear(x.squeeze(2).transpose(-2,-1), self.qkv.weight.squeeze()).transpose(-2,-1).unsqueeze(2)
        else:
            qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)
        q = q.view(b, self.config.num_heads, self.dim_head, s)
        k = k.view(b, self.config.num_heads, self.dim_head, s)
        v = v.view(b, self.config.num_heads, self.dim_head, s)

        # RoPE
        cos, sin = self.rotary_emb(x, position_ids=position_ids)
        cos, sin = cos.transpose(-1,-2), sin.transpose(-1 ,-2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Switch between global or local attention as appropriate.
        mask = attention_mask if self.use_global_attention else self._sliding_window_mask(attention_mask)

        attn = self.original_attn(q, k, v, mask, self.config.num_heads, self.dim_head)
        return self.out(attn)

    @staticmethod
    def original_attn(q, k, v, mask, heads, dim_head):
        bs = q.size(0)
        mh_q = q.view(bs, heads, dim_head, -1)
        mh_k = k.view(bs, heads, dim_head, -1)
        mh_v = v.view(bs, heads, dim_head, -1)

        attn_weights = torch.einsum("bhcq,bhck->bhqk", [mh_q, mh_k])
        attn_weights.mul_(dim_head**-0.5)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = attn_weights.softmax(dim=3)

        attn = torch.einsum("bhqk,bhck->bhcq", [attn_weights, mh_v])
        return attn.contiguous().view(bs, heads * dim_head, 1, -1)

    def _sliding_window_mask(self, global_attention_mask):
        # TODO: Compute this more efficiently. It's stored as a boolean tensor in CoreML: ~66MB for 8192 sequence length.
        seq_length = global_attention_mask.shape[2]
        position_indices = torch.arange(seq_length).unsqueeze(0)
        distances = torch.abs(position_indices - position_indices.T)

        # 1 for positions within window, 0 outside
        window_mask = (
            (distances <= self.config.local_attention_window_size // 2).unsqueeze(0).unsqueeze(0).to(global_attention_mask.device)
        )

        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), self.mask_min_value)
        return sliding_window_mask

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.up_gate = nn.Conv2d(config.hidden_size, config.intermediate_size * 2, kernel_size=1, bias=False)
        self.act = torch.nn.GELU()
        self.down = nn.Conv2d(config.intermediate_size, config.hidden_size, kernel_size=1, bias=False)

    def forward(self, x):
        x, gate = self.up_gate(x).chunk(2, dim=1)
        return self.down(self.act(x) * gate)

class Block(nn.Module):
    def __init__(self, config: Config, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.pre_attn_norm = LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias) if layer_index != 0 else nn.Identity()
        self.attn = Attention(config, layer_index)
        self.pre_mlp_norm = LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = MLP(config)

    def forward(self, x, position_ids, attention_mask):
        # print(f"{self.layer_index} abs max:", x.abs().max())
        # print(f"{self.layer_index} seq:", x.abs().topk(3, dim=-1).indices, "\nch:", x.abs().topk(3, dim=-3).indices)
        x = x + self.attn(self.pre_attn_norm(x), position_ids, attention_mask)
        return x + self.mlp(self.pre_mlp_norm(x))

class MaskedLMHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dense = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1, bias=False)
        self.act = torch.nn.GELU()
        self.norm = LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.decoder = nn.Conv2d(config.hidden_size, config.vocab_size, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.norm(self.act(self.dense(x)))
        return self.decoder(x)

class Model(nn.Module):
    def __init__(self, config: Config, head: Optional[nn.Module]=None):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])
        self.ln_f = LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.head = head(config) if head else nn.Identity()

    def forward(self, x, attention_mask):
        x = self.embeddings(x)
        x = x.transpose(-1,-2).unsqueeze(2) # to bc1s
        position_ids = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids, attention_mask)
        x = self.ln_f(x)
        return self.head(x) # MaskedLM, Classification, etc. or no-op

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, head=None):
        overrides = {k:v for k,v in {
            "base": dict(hidden_size=768, intermediate_size=1152, num_layers=22, num_heads=12),
            "large": dict(hidden_size=1024, intermediate_size=2624, num_layers=28, num_heads=16),
        }.items() if k in model_name_or_path}
        assert len(overrides) == 1, f"Only {list(overrides.keys())} models are supported."
        overrides = list(overrides.values())[0]
        config = Config(**overrides)

        from transformers import AutoModelForMaskedLM
        hf_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        model = Model(config, head=head)

        hf_sd = hf_model.state_dict()
        sd = model.state_dict()

        sd["embeddings.embeddings.weight"].copy_(hf_sd["model.embeddings.tok_embeddings.weight"])
        sd["embeddings.norm.weight"].copy_(hf_sd["model.embeddings.norm.weight"])

        for i in range(config.num_layers):
            # Linear -> Conv2d.
            sd[f"layers.{i}.attn.qkv.weight"].copy_(hf_sd[f"model.layers.{i}.attn.Wqkv.weight"].unsqueeze(-1).unsqueeze(-1))
            sd[f"layers.{i}.attn.out.weight"].copy_(hf_sd[f"model.layers.{i}.attn.Wo.weight"].unsqueeze(-1).unsqueeze(-1))
            sd[f"layers.{i}.mlp.up_gate.weight"].copy_(hf_sd[f"model.layers.{i}.mlp.Wi.weight"].unsqueeze(-1).unsqueeze(-1))
            sd[f"layers.{i}.mlp.down.weight"].copy_(hf_sd[f"model.layers.{i}.mlp.Wo.weight"].unsqueeze(-1).unsqueeze(-1))

            # LayerNorm
            if i != 0:
                sd[f"layers.{i}.pre_attn_norm.weight"].copy_(hf_sd[f"model.layers.{i}.attn_norm.weight"])
            sd[f"layers.{i}.pre_mlp_norm.weight"].copy_(hf_sd[f"model.layers.{i}.mlp_norm.weight"])

        sd["ln_f.weight"].copy_(hf_sd["model.final_norm.weight"])

        if head:
            sd["head.dense.weight"].copy_(hf_sd["head.dense.weight"].unsqueeze(-1).unsqueeze(-1))
            sd["head.norm.weight"].copy_(hf_sd["head.norm.weight"])
            sd["head.decoder.weight"].copy_(hf_sd["model.embeddings.tok_embeddings.weight"].unsqueeze(-1).unsqueeze(-1))
            sd["head.decoder.bias"].copy_(hf_sd["decoder.bias"])

        return model

class MaskedLMModel:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        return Model.from_pretrained(model_name_or_path, head=MaskedLMHead)

class LayerNorm(nn.Module):
    def __init__(self, num_channels,eps=1e-5,elementwise_affine=True, bias=True):
        super().__init__()
        self.expected_rank = len('BC1S')

        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels) if bias else torch.zeros(num_channels))

    def forward(self, inputs):
        assert inputs.size(1) == self.num_channels
        # Match the reference implementation so that coremltools can use a single op.
        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)
                   ) * self.weight.view(1, self.num_channels, 1, 1)

        return out

    def __repr__(self):
        return f'LayerNorm(({self.num_channels},), eps={self.eps}, elementwise_affine={self.elementwise_affine})'

# RoPE -- Not gonna re-implement this. Straight from HF, minimal modifications for BC1S tensor shape.

class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    # Modified for BC1S tensor shape.
    x1 = x[:, :, : x.shape[-2] // 2, :]  # (B, nh, hs/2, T)
    x2 = x[:, :, x.shape[-2] // 2 :, :]  # (B, nh, hs/2, T)
    return torch.cat((-x2, x1), dim=-2)  # (B, nh, hs, T)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
