# ModernBERT for Apple Neural Engine

ModernBERT model optimized for Apple Neural Engine.

> 🏎️ 3.0 TFLOP/s (1024 token input; base model)
>
> 🔋 2.1 W of power (1024 token input; base model)
>
> 🤏 1 file for the model definition (a la nanoGPT)

# Install
```shell
$ python -m venv env
$ . env/bin/activate
$ pip install -r requirements.txt
```

# Convert to CoreML
```shell
$ python convert.py
$ python predict.py $path_to_model.mlpackage "The sky is [MASK]."
```

# Compare to 🤗
Compare accuracy of models to HuggingFace's implementation.
```shell
# Compare the PyTorch model in model.py
$ python diff_torch.py
# Compare a converted CoreML model
$ python diff_coreml.py $path_to_model.mlpackage
```

# A Note on Precision
The Neural Engine requires float16 weights and activations. Some computations can be performed in float32, but outlier activations can still severely degrade output predictions.

ModernBERT, like other modern decoder-only LLMs, exhibits outlier activations on the order of 20-30k. Without intervention these are enough to visibly impact the CoreML model's predictions on the Neural Engine.

To mitigate this, the model conversion process in this repo uses QuaRot/SpinQuant-style orthogonal rotations. This greatly improves the model's fidelity (as measured by the KL divergence). However token predictions will not exactly match a PyTorch model that does some/all computation in a higher precision (bfloat16, float32). Be sure to test for your use case.

# Credits
Borrows heavily from:
- [transformers](https://github.com/huggingface/transformers/blob/f42084e6411c39b74309af4a7d6ed640c01a4c9e/src/transformers/models/modernbert/modeling_modernbert.py#L822)
- [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [QuaRot](https://github.com/spcl/QuaRot)

# Areas for Improvement
- support longer sequence lengths (> 1024)
  - alternative attention implementations (split einsum, efficient attention for longer sequence length)
- generate/use [SpinQuant](https://github.com/facebookresearch/SpinQuant) matrices for improved outlier reduction
- investigate [PrefixQuant](https://github.com/ChenMnZ/PrefixQuant) for improved outlier reduction
- convert core model separately from heads to allow hot-swapping of different heads
- pack short sequences into single prediction
- support for heads beyond masked LM
- enumerated shapes for inputs [see](https://github.com/0seba/ModernBERT-AppleNeuralEngine/commit/46b73ba40fbeb712f1c47b084922190c3058ce29)
