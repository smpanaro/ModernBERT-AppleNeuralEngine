import sys
import torch
from torch.nn import functional as F
import numpy as np
import coremltools as ct
from model import Model, MaskedLMModel

"""
Convert a ModernBERT HuggingFace model to CoreML.
"""

torch.set_grad_enabled(False)

model_name_or_path = "answerdotai/ModernBERT-base"
max_seq_len = 1024
if len(sys.argv) == 3:
    model_name_or_path = sys.argv[1]
    max_seq_len = int(sys.argv[2])
elif len(sys.argv) == 2 and sys.argv[1].isnumeric():
    max_seq_len = int(sys.argv[1])
elif len(sys.argv) == 2:
    model_name_or_path = sys.argv[1]
else:
    assert False, f"Usage: {sys.argv[0]} model_name_or_path [max_seq_len]"

print(f"Converting {model_name_or_path} to CoreML...")
model = MaskedLMModel.from_pretrained(model_name_or_path).eval()
model.rotate()

input_ids = torch.zeros( (1, max_seq_len), dtype=torch.int)
input_ids[..., :] = 50283 # PAD
seq = torch.tensor([50281,510,5347,273,6181,310,50284,15,50282], dtype=torch.int)
input_ids[..., :seq.shape[-1]] = seq
mask = torch.zeros((1,max_seq_len,1,max_seq_len))
mask[:,seq.shape[-1]:,:,:] = -1e4
mask[:,:,:,seq.shape[-1]:] = -1e4

output_name = "hidden_states" if isinstance(model, MaskedLMModel) else "logits"

mlmodel= ct.convert(
    torch.jit.trace(model, (input_ids, mask)),
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="mask", shape=mask.shape, dtype=np.float16, default_value=np.zeros_like(mask).astype(np.float16)),
    ],
    outputs=[
        ct.TensorType(name=output_name),
    ],
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT16,
    # For initial prediction:
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
assert isinstance(mlmodel, ct.models.MLModel), "unexpected converted model type"

input_output_descriptions = {
    "input_ids": "Indices of input sequence tokens in the vocabulary",
    "mask": "Mask for defining which tokens should attend to each other. 0 means attend and large negative number (e.g. -1e4) means do not attend.",
    "hidden_states": "Raw outputs from the model. Typically further processed by a task-specific head.",
    "logits": "Un-normalized per-token predictions.",
}

for k in mlmodel.input_description:
    mlmodel.input_description[k] = input_output_descriptions[k]
for k in mlmodel.output_description:
    mlmodel.output_description[k] = input_output_descriptions[k]

mlmodel.user_defined_metadata["Source Model"] = model_name_or_path

mlmodel.save(f"{model_name_or_path.replace('/', '-')}-{max_seq_len}.mlpackage")

model = MaskedLMModel.from_pretrained(model_name_or_path).eval() # Reload non-rotated model.
coreml_out = torch.from_numpy(mlmodel.predict({"input_ids": input_ids.numpy(), "mask": mask.numpy()})[output_name])
torch_out = model(input_ids, mask)
# Sometime useful for debugging.
# print("CoreML Top 4\n", coreml_out.topk(4, dim=1))
# print("Torch Top 4", torch_out.topk(4, dim=1))
# print("CoreML<>Torch max absolute difference:", (coreml_out - torch_out).abs().max())

kl = F.kl_div(F.log_softmax(coreml_out[...,:seq.shape[-1]], dim=1), F.log_softmax(torch_out[...,:seq.shape[-1]], dim=1), log_target=True, reduction='batchmean')
print("CoreML<>Torch KL divergence:", kl) # smaller is better
