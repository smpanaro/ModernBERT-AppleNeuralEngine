import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
import math

"""
nanoGPT-inspired, pared down version of the HF ModernGPT implementation,
optimized for Apple Neural Engine a la ml-ane-transformers + ml-stable-diffusion.
optional support for QuaRot/SpinQuant-style outlier activation reduction.
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
        self.norm = LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embeddings(input_ids) # ~570us on CPU for 512 tokens.
        if self.config.match_hf and isinstance(self.norm, LayerNorm):
            # bc1s LayerNorm introduces a slight numerical accuracy drift.
            return F.layer_norm(x, (x.size()[-1],), self.norm.weight, self.norm.bias, self.norm.eps).transpose(-1,-2).unsqueeze(2)
        x = x.transpose(-1,-2).unsqueeze(2) # to bc1s
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, config: Config, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.qkv = nn.Conv2d(config.hidden_size, config.hidden_size * 3, kernel_size=1, bias=False)
        self.dim_head = config.hidden_size // config.num_heads

        self.mask_min_value = -1e4 # torch.finfo(attention_mask.dtype).min is too small for ANE / float16.

        self.use_global_attention = self.layer_index % self.config.global_attn_every_n_layers == 0
        rope_theta = config.global_rope_theta if self.use_global_attention else config.local_rope_theta
        self.rotary_emb = ModernBertRotaryEmbedding(dim=self.dim_head, base=rope_theta)

        self.out = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1, bias=False)

    def forward(self, x, position_ids, attention_mask, sliding_window_mask=None):
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
        mask = self._attention_mask(attention_mask, sliding_window_mask)

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

    def _attention_mask(self, global_attention_mask, precomputed_sliding_mask=None):
        if self.use_global_attention:
            return global_attention_mask
        if precomputed_sliding_mask is not None:
            return precomputed_sliding_mask
        return self.sliding_window_mask(self.config, global_attention_mask, self.mask_min_value)

    @staticmethod
    def sliding_window_mask(config: Config, global_attention_mask, mask_min_value: float = -1e4):
        # TODO: Compute this more efficiently. It's stored as a boolean tensor in CoreML: ~66MB for 8192 sequence length.
        # ~252us on CPU for 512 tokens.
        seq_length = global_attention_mask.shape[2]
        position_indices = torch.arange(seq_length).unsqueeze(0)
        distances = torch.abs(position_indices - position_indices.T)

        # 1 for positions within window, 0 outside
        window_mask = (
            (distances <= config.local_attention_window_size // 2).unsqueeze(0).unsqueeze(0).to(global_attention_mask.device)
        )

        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), mask_min_value)
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
        # Optional transform for residual connection. Useful if applying QuaRot-style rotations.
        self.residual_transform = nn.Identity()

    def forward(self, x, position_ids, attention_mask, sliding_window_mask=None):
        x = self.residual_transform(x) + self.attn(self.pre_attn_norm(x), position_ids, attention_mask, sliding_window_mask)
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
        self.unrotate = nn.Identity()

    def forward(self, x, attention_mask):
        sliding_window_mask = Attention.sliding_window_mask(self.config, attention_mask) # Do all CPU work first.
        x = self.embeddings(x)
        position_ids = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids, attention_mask, sliding_window_mask)
        x = self.ln_f(x)
        x = self.unrotate(x)
        return self.head(x) # MaskedLM, Classification, etc. or no-op

    def rotate(self):
        rotate_model(self)

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
    # From ml-stable-diffusion
    def __init__(self, num_channels,eps=1e-5,elementwise_affine=True, bias=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels)) if bias else None

    def forward(self, inputs):
        assert inputs.size(1) == self.num_channels
        # Match the reference implementation so that
        # coremltools recognizes it and replaces it
        # with a single MIL op.
        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom

        if self.elementwise_affine:
            if self.bias is not None:
                out = (out + self.bias.view(1, self.num_channels, 1, 1)
                    ) * self.weight.view(1, self.num_channels, 1, 1)
            else:
                out = out * self.weight.view(1, self.num_channels, 1, 1)

        return out

    def __repr__(self):
        return f'LayerNorm(({self.num_channels},), eps={self.eps}, elementwise_affine={self.elementwise_affine})'

class RMSNorm(nn.Module):
    """
    ANE-friendly RMSNorm
    """
    def __init__(self, size: int, dim: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((1,size,1,1)))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # This is equivalent to simpler RMSNorm implementations but it
        # uses linalg.norm which is resilient to overflow when converted for ANE.
        # Adding the eps is important for numerical equivalence.
        # Clamping as an alternative to eps seems to trigger a bug in ANE, hence this approach.
        # Derivation of the eps channel value:
        # mean = x0^2 / C + x1^2 / C + ... + xN^2 / C
        # sqrt(mean + eps) = sqrt(x0^2 / C + x1^2 / C + ... + xN^2 / C + eps*C/C)
        #               = sqrt(x0^2 + x1^2 + ... + xN^2 + eps*C) / sqrt(C)
        eps_chan = torch.ones((x.size(0), 1, x.size(2), x.size(3))) * ((self.eps*x.size(1)) ** 0.5)
        x_eps = torch.cat((x, eps_chan), dim=1)

        norm_x = torch.linalg.norm(x_eps, dim=1, keepdim=True)
        x_normed = x / norm_x
        x_normed = x_normed * math.sqrt(x.size(1))

        x_normed = x_normed.to(dtype=dtype)
        return x_normed * self.weight

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

# Orthogonal Rotation
# Apply QuaRot orthogonal rotations to the model to help reduce activation outliers
# that make it difficult to run the model accurately in float16 (required for ANE).

def rotate_model(model: Model):
    fuse_embeddings(model)
    fuse_layer_norms(model)
    # Q = get_orthogonal_matrix(model.config.hidden_size, model.embeddings.embeddings.weight.device)
    Q = random_hadamard_matrix(model.config.hidden_size, model.embeddings.embeddings.weight.device)
    rotate_embeddings(model, Q)
    rotate_layers(model, Q)
    rotate_head(model, Q)

def fuse_embeddings(model):
    for W in [model.embeddings.embeddings]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

def fuse_layer_norms(model):
    def fuse_ln_conv(layernorm, conv_layers):
        assert not isinstance(layernorm, nn.Identity)

        for conv in conv_layers:
            conv_dtype = conv.weight.dtype
            W_ = conv.weight.data.squeeze().double()
            conv.weight.data = (W_ * layernorm.weight.double()).unsqueeze(-1).unsqueeze(-1).to(conv_dtype)

            if hasattr(layernorm, 'bias') and layernorm.bias is not None:
                if conv.bias is None:
                    conv.bias = nn.Parameter(torch.zeros(conv.out_channels, dtype=torch.float64))
                conv.bias.data = conv.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
                conv.bias.data = conv.bias.data.to(conv_dtype)

    def bake_mean_into_conv(conv: nn.Conv2d) -> None:
        """
        This function takes a conv layer and subtracts the means from the
        weights and biases. This will result in the conv layer performing
        the mean substitution which is usually done inside layernorm.
        """
        conv_dtype = conv.weight.dtype
        W_ = conv.weight.data.squeeze().double()
        conv.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).unsqueeze(-1).unsqueeze(-1).to(conv_dtype)
        if conv.bias is not None:
            b_ = conv.bias.data.double()
            conv.bias.data = b_ - b_.mean()
            conv.bias.data = conv.bias.data.to(conv_dtype)

    def equivalent_rms_norm(layernorm):
        assert not isinstance(layernorm, nn.Identity)
        nm = RMSNorm(layernorm.weight.size(0), eps=layernorm.eps)
        return nm

    # Slight difference from QuaRot here (and in rotate_layers) to account
    # for the fact that the first layer norm is immediately after the
    # embeddings in ModernBERT, unlike Llama where it is immediately
    # before the first QKV projection.
    scale_matrix = torch.diag(model.embeddings.norm.weight).double()
    centering_matrix = torch.eye(model.config.hidden_size, dtype=torch.float64) - (1./model.config.hidden_size)
    model.layers[0].residual_transform = nn.Conv2d(model.config.hidden_size, model.config.hidden_size, kernel_size=1, bias=False)
    model.layers[0].residual_transform.weight.data = (centering_matrix @ scale_matrix).float().unsqueeze(-1).unsqueeze(-1)

    fuse_ln_conv(model.embeddings.norm, [model.layers[0].attn.qkv])
    model.embeddings.norm = equivalent_rms_norm(model.embeddings.norm)

    for idx, layer in enumerate(model.layers):
        if not isinstance(layer.pre_attn_norm, nn.Identity):
            fuse_ln_conv(layer.pre_attn_norm, [layer.attn.qkv])
        fuse_ln_conv(layer.pre_mlp_norm, [layer.mlp.up_gate])

        if not isinstance(layer.pre_attn_norm, nn.Identity):
            layer.pre_attn_norm = equivalent_rms_norm(layer.pre_attn_norm)

        layer.pre_mlp_norm = equivalent_rms_norm(layer.pre_mlp_norm)
        bake_mean_into_conv(layer.attn.out)
        bake_mean_into_conv(layer.mlp.down)

    if isinstance(model.unrotate, nn.Identity):
        model.unrotate = nn.Conv2d(model.config.hidden_size, model.config.hidden_size, kernel_size=1, bias=False)
        identity = torch.eye(model.config.hidden_size, dtype=model.unrotate.weight.dtype, device=model.unrotate.weight.device)
        model.unrotate.weight.data = identity.unsqueeze(-1).unsqueeze(-1)
    fuse_ln_conv(model.ln_f, [model.unrotate])
    model.ln_f = equivalent_rms_norm(model.ln_f)

def rotate_embeddings(model, Q):
    W = model.embeddings.embeddings
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=Q.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_layers(model, Q):
    for idx, layer in enumerate(model.layers):
        if idx == 0:
            # See comment about the first LayerNorm in fuse_layer_norms.
            layer.residual_transform.weight.data = (Q.T @ layer.residual_transform.weight.double().squeeze() @ Q).float().unsqueeze(-1).unsqueeze(-1)
        rotate_attention(layer.attn, Q)
        rotate_mlp(layer.mlp, Q)

def rotate_attention(attn, Q):
    for W in [attn.qkv, attn.out]:
        dtype = W.weight.dtype
        W_ = W.weight.squeeze().to(device=Q.device, dtype=torch.float64)
        if W == attn.qkv:
            W.weight.data = torch.matmul(W_, Q).unsqueeze(-1).unsqueeze(-1).to(device="cpu", dtype=dtype)
        else:  # out projection
            W.weight.data = torch.matmul(Q.T, W_).unsqueeze(-1).unsqueeze(-1).to(device="cpu", dtype=dtype)

def rotate_mlp(mlp, Q):
    dtype = mlp.up_gate.weight.dtype
    W_ = mlp.up_gate.weight.squeeze().to(device=Q.device, dtype=torch.float64)
    mlp.up_gate.weight.data = torch.matmul(W_, Q).unsqueeze(-1).unsqueeze(-1).to(device="cpu", dtype=dtype)

    dtype = mlp.down.weight.dtype
    W_ = mlp.down.weight.squeeze().to(device=Q.device, dtype=torch.float64)
    mlp.down.weight.data = torch.matmul(Q.T, W_).unsqueeze(-1).unsqueeze(-1).to(device="cpu", dtype=dtype)

def rotate_head(model, Q):
    W = model.unrotate
    assert not isinstance(W, nn.Identity), "fusing requires that we have already updated this to be an identity matrix"
    dtype = W.weight.dtype
    W_ = W.weight.data.squeeze().to(device=Q.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).unsqueeze(-1).unsqueeze(-1).to(device="cpu", dtype=dtype)

def get_orthogonal_matrix(size, device):
    random_matrix = torch.randn(size, size, dtype=torch.float64, device=device)
    q, r = torch.linalg.qr(random_matrix)
    return q * torch.sign(torch.diag(r)).unsqueeze(0)

# Hadamard Orthogonal Matrix
# Subset of https://github.com/spcl/QuaRot/blob/main/fake_quant/hadamard_utils.py
# Should be better than a standard orthogonal matrix.

def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)

def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 12 == 0: # ModernBERT-base (768)
        assert (is_pow2(n // 12))
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else: # ModernBERT-large (1024)
        assert (is_pow2(n)), f"n: {n}"
        K = 1

    return hadK, K

def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

# hadamard matrices for had12, had36.pal2, had52,will,
# # had60.pal, had108.pal, had140.pal, had156.will, had172.will:
# http://www.neilsloane.com/hadamard/index.html
def get_had12():
    return torch.FloatTensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
        [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
        [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
        [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
    ])
