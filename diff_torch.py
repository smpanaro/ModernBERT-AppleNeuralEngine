from transformers import AutoTokenizer, AutoModelForMaskedLM
from model import Model, MaskedLMHead
import torch
from torch import nn
from torch.nn import functional as F
import sys

"""
Perform a module-by-module comparison of this repo's PyTorch ModernBERT re-implementation
and the official HuggingFace implementation, either with or without outlier-reducing rotations.
"""

torch.set_grad_enabled(False)

do_rotate = len(sys.argv) > 1 and sys.argv[1] == "rotate"

def print_equal(name, is_equal=None, max_delta=None):
    green_text = "\033[32m"
    red_text = "\033[31m"
    reset = "\033[0m"
    if is_equal is not None:
        result = f"{green_text}equal{reset}" if is_equal else f"{red_text}not equal{reset}"
    elif max_delta is not None:
        result = f"±{max_delta.item(): .4e}"
    else:
        result = f"{red_text}missing{reset}"
    print(f"{name:>18}: {result}")

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"comparing {model_id} to 🤗")

hf_masked_model = AutoModelForMaskedLM.from_pretrained(model_id)
hf_model = hf_masked_model.model
hf_model.config._attn_implementation = "eager"

ne_model = Model.from_pretrained(model_id, head=MaskedLMHead)
ne_head = ne_model.head
ne_model.head = nn.Identity()
ne_model.config.match_hf = True

if do_rotate:
    print("applying orthogonal rotation to minimize outliers")
    print("⚠️ expect results to be close but not exactly equal")
    ne_model.rotate() # Reduce outlier activations.

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

def to_bsc(x):
    assert x.shape[2] == 1
    return x.transpose(1,3).squeeze(2)

def to_bc1s(x):
    assert len(x.shape) == 3
    return x.unsqueeze(2).transpose(1,3)

hf_emb = hf_model.embeddings(inputs["input_ids"])
ne_emb = ne_model.embeddings(inputs["input_ids"])
# assert torch.equal(hf_emb, ne_emb), "embeddings not equal"
print_equal("embeddings", torch.equal(hf_emb, to_bsc(ne_emb)))

position_ids = torch.arange(hf_emb.shape[1], device=hf_emb.device).unsqueeze(0)

# Attention Masks
hf_base_mask = inputs["attention_mask"] # (batch, seq), int
hf_base_mask[..., -1] = 0
hf_attention_mask, hf_sliding_mask = hf_model._update_attention_mask(inputs["attention_mask"], output_attentions=False)
hf_attention_mask.clamp_(min=ne_model.layers[0].attn.mask_min_value)
hf_sliding_mask.clamp_(min=ne_model.layers[0].attn.mask_min_value)
ne_attention_mask = torch.zeros((1,1,hf_emb.shape[1], hf_emb.shape[1]))
ne_attention_mask[..., -1] = ne_model.layers[0].attn.mask_min_value
# assert torch.equal(hf_attention_mask, ne_attention_mask)
print_equal("attn mask", torch.equal(hf_attention_mask, ne_attention_mask))
ne_attention_mask = ne_attention_mask.permute(0,3,1,2) # 11qk -> bk1q

# Global Rotary Emb
hf_rotary_emb_0 = hf_model.layers[0].attn.rotary_emb(hf_emb, position_ids=position_ids)
ne_rotary_emb_0 = ne_model.layers[0].attn.rotary_emb(ne_emb, position_ids=position_ids)
# assert torch.equal(hf_rotary_emb_0[0], ne_rotary_emb_0[0])
print_equal("RoPE global", torch.equal(hf_rotary_emb_0[0], ne_rotary_emb_0[0]))

# Global Attention
hf_attn_0 = hf_model.layers[0].attn(hf_emb, attention_mask=hf_attention_mask, sliding_window_mask=hf_sliding_mask, position_ids=position_ids)
ne_attn_0 = ne_model.layers[0].attn(to_bc1s(hf_emb), position_ids, ne_attention_mask)
# print((hf_attn_0[0] - to_bsc(ne_attn_0)).abs().max())
# assert torch.equal(hf_attn_0[0], to_bsc(ne_attn_0)), "attention 0 not equal"
print_equal("attention global", max_delta=(hf_attn_0[0] - to_bsc(ne_attn_0)).abs().max())

# Local Rotary Emb
hf_rotary_emb_1 = hf_model.layers[1].attn.rotary_emb(hf_emb, position_ids=position_ids)
ne_rotary_emb_1 = ne_model.layers[1].attn.rotary_emb(ne_emb, position_ids=position_ids)
# assert torch.equal(hf_rotary_emb_1[0], ne_rotary_emb_1[0])
print_equal("RoPE local", torch.equal(hf_rotary_emb_1[0], ne_rotary_emb_1[0]))

# Local Sliding Mask
ne_sliding_mask_1 = ne_model.layers[1].attn.sliding_window_mask(ne_model.layers[1].attn.config, ne_attention_mask)
# assert torch.equal(hf_sliding_mask, ne_sliding_mask_1)
print_equal("sliding mask", torch.equal(hf_sliding_mask, ne_sliding_mask_1.permute(0,2,3,1)))

# Local Attention
hf_attn_1 = hf_model.layers[1].attn(hf_emb, attention_mask=hf_attention_mask, sliding_window_mask=hf_sliding_mask, position_ids=position_ids)
ne_attn_1 = ne_model.layers[1].attn(to_bc1s(hf_emb), position_ids, ne_attention_mask)
# assert torch.equal(hf_attn_1[0], to_bsc(ne_attn_1)), "attention 1 not equal"
# print((hf_attn_1[0] - to_bsc(ne_attn_1)).abs().max())
print_equal("attention local", max_delta=(hf_attn_1[0] - to_bsc(ne_attn_1)).abs().max())

# MLP
hf_mlp = hf_model.layers[0].mlp(hf_attn_0[0])
ne_mlp = ne_model.layers[0].mlp(to_bc1s(hf_attn_0[0]))
# assert torch.equal(hf_mlp, to_bsc(ne_mlp)), "mlp not equal"
# print((hf_mlp - to_bsc(ne_mlp)).abs().max())
print_equal("mlp", torch.equal(hf_mlp, to_bsc(ne_mlp)))

# Block 0 (no pre-attention norm)
hf_block_0 = hf_model.layers[0](hf_emb, attention_mask=hf_attention_mask, sliding_window_mask=hf_sliding_mask, position_ids=position_ids)
ne_block_0 = ne_model.layers[0](ne_emb, position_ids, ne_attention_mask)
# assert torch.equal(hf_block_0[0], to_bsc(ne_block_0)), "block 0 not equal"
print_equal("block w/o norm", max_delta=(hf_block_0[0] - to_bsc(ne_block_0)).abs().max())

# Block 1 (with pre-attention norm)
hf_block_1 = hf_model.layers[1](hf_block_0[0], attention_mask=hf_attention_mask, sliding_window_mask=hf_sliding_mask, position_ids=position_ids)
ne_block_1 = ne_model.layers[1](ne_block_0, position_ids, ne_attention_mask)
# assert torch.equal(hf_block_1[0], to_bsc(ne_block_1)), "block 1 not equal"
print_equal("block w/norm", max_delta=(hf_block_1[0] - to_bsc(ne_block_1)).abs().max())

# Model
# print(inputs["input_ids"], ne_attention_mask)
hf_model._update_attention_mask = lambda *args, **kwargs: (hf_attention_mask, hf_sliding_mask)
hf_state = hf_model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
ne_state = ne_model(inputs["input_ids"], ne_attention_mask)
# assert torch.equal(hf_logits.last_hidden_state, to_bsc(ne_logits))
print_equal("model headless", max_delta=(hf_state - to_bsc(ne_state)).abs().max())

# Mask LM Model
ne_model.head = ne_head
hf_logits = hf_masked_model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
ne_logits = ne_model(inputs["input_ids"], ne_attention_mask)
print_equal("model masked LM", max_delta=(hf_logits - to_bsc(ne_logits)).abs().max())
kl = F.kl_div(F.log_softmax(to_bsc(ne_logits), dim=-1).double(), F.log_softmax(hf_logits, dim=-1).double(), log_target=True, reduction='batchmean')
print_equal("kl div", max_delta=kl)
mask_indices = inputs["input_ids"] == tokenizer.mask_token_id
for top_k in [1, 4, 8, 80]:
    print_equal(f"top {top_k} masked indices",
        torch.equal(hf_logits[mask_indices, :].topk(top_k).indices,
            to_bsc(ne_logits)[mask_indices, :].topk(top_k).indices))

if not do_rotate:
    print("-> re-run with 'rotate' argument to compare with an orthogonally rotated model")
