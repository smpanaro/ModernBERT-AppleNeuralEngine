from transformers import AutoTokenizer, AutoModelForMaskedLM
import coremltools as ct
import torch
from torch.nn import functional as F
import sys

"""
Compare the output logits of a ModernBERT CoreML model to the HuggingFace model.
"""

assert len(sys.argv) in [2, 3], f"Usage: {sys.argv[0]} path_to_mlpackage [masked_text]"
model_path = sys.argv[1]

model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
sequence_length = {x.name: x for x in model.get_spec().description.input}["input_ids"].type.multiArrayType.shape[-1]
model_id = model.get_spec().description.metadata.userDefined["Source Model"]
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForMaskedLM.from_pretrained(model_id).eval()

text = "The capital of France is [MASK]." if len(sys.argv) == 2 else sys.argv[2]

inputs = tokenizer(text, padding='max_length', max_length=sequence_length, return_tensors="pt")
first_pad_index = torch.where(inputs["input_ids"] == tokenizer.pad_token_id)[1][0].item()
mask = torch.zeros((1, 1, sequence_length, sequence_length))
mask[..., :, first_pad_index:] = -1e4
mask[..., first_pad_index:, :] = -1e4

outputs = model.predict({"input_ids": inputs["input_ids"].int().numpy(), "mask": mask.numpy()})
logits = list(outputs.values())[0]

logits = torch.from_numpy(logits).squeeze(2).transpose(-1,-2)
hf_logits = hf_model(**inputs).logits

logits = logits[:, :first_pad_index, :]
hf_logits = hf_logits[:, :first_pad_index, :]
kl = F.kl_div(F.log_softmax(logits, dim=-1), F.log_softmax(hf_logits, dim=-1), log_target=True, reduction='batchmean')
print("\nKL Divergence")
print("Sequence only (excl. padding):", kl.item())
