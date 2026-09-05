# llama.cpp patches

Project-specific changes to the pinned `llama.cpp` submodule. Apply them in
filename order after initializing the submodule:

```sh
git submodule update --init llama.cpp
scripts/apply-llama-patches.sh
```

`scripts/configure.sh` and S2S/NMT Docker builds apply the series
automatically.

## Patches

- **0001-batch-all-recurrent-outputs.patch** - keeps equal-length recurrent
  sequences in one microbatch when all token outputs are requested.
- **0002-enable-nvfp4-quantization.patch** - enables NVFP4 model quantization
  and its Q8 fallback path.
- **0003-gemma3-attention-scale.patch** - honors the attention scale stored in
  Gemma 3 model metadata, with the upstream default as a fallback.
- **0004-mamba2-flat-projections.patch** - keeps Mamba2 input and output
  projections flat so batched CUDA decode dispatches GEMM instead of GEMV.
  Backports [llama.cpp PR #27513](https://github.com/ggml-org/llama.cpp/pull/27513).
