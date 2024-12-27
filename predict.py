from transformers import AutoTokenizer
import coremltools as ct
import torch
import sys

"""
Predict a masked token using a ModernBERT CoreML model.
"""

assert len(sys.argv) in [2, 3], f"Usage: {sys.argv[0]} path_to_mlpackage [masked_text]"
model_path = sys.argv[1]

model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
sequence_length = {x.name: x for x in model.get_spec().description.input}["input_ids"].type.multiArrayType.shape[-1]
model_id = model.get_spec().description.metadata.userDefined["Source Model"]
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "The capital of France is [MASK]." if len(sys.argv) == 2 else sys.argv[2]

inputs = tokenizer(text, padding='max_length', max_length=sequence_length, return_tensors="pt")
first_pad_index = torch.where(inputs["input_ids"] == tokenizer.pad_token_id)[1][0].item()
mask = torch.zeros((1, 1, sequence_length, sequence_length))
mask[..., :, first_pad_index:] = -1e4
mask[..., first_pad_index:, :] = -1e4

outputs = model.predict({"input_ids": inputs["input_ids"].int().numpy(), "mask": mask.numpy()})
logits = list(outputs.values())[0]

masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
# logits are bc1s (not bsc)
logits = torch.from_numpy(logits).softmax(dim=1) # to probs
topk = logits[..., masked_index].topk(5, dim=1)
print("\nText:", text)
print("Predictions")
for i in range(topk.indices.shape[-2]):
    token_id = topk.indices[0, i]
    prob = topk.values[0, i].item()
    token = tokenizer.decode(token_id)
    print(f"{prob:.4f}: {token}")
