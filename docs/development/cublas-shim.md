# cuBLAS shim and GPU kernels

The minimal runtime image ships **no NVIDIA cuBLAS**. ggml-cuda's
non-quantized GEMMs (FastConformer attention, subsampling convs, CTC head; the
q8/RNNT weight matmuls already use ggml's quantized kernels) are served instead
by an in-tree drop-in `libcublas.so.13`.

## The shim

`kernels/cublas_shim.cu` (with the symbol map `kernels/ver_cublas.map`) is a
drop-in `libcublas.so.13`: shape-specialized CUDA GEMM/GEMV kernels, including
WMMA tensor-core paths, but **no cuBLASLt**. It inherits
`CMAKE_CUDA_ARCHITECTURES` when set and falls back to JIT-portable `compute_80`
PTX for ad-hoc builds. Dropping real cuBLAS + cuBLASLt is the bulk of the
container size. The shim is built as a separate library from ggml.

It's a normal CMake target, **`NEMO_SPEECH_CUBLAS_SHIM` (default `ON`)**, built
whenever `GGML_CUDA` is also on (Linux only, auto-skipped on Windows; a no-op
for Metal/Vulkan/CPU builds). So a CUDA
build produces `libcublas.so.13` in its binary directory by default; the
Dockerfile relies on that and skips real cuBLAS/cuBLASLt in the library closure.
To run a local CLI command with the shim, put it first on the loader path:

```bash
scripts/configure.sh cuda-asr
cmake --build --preset cuda-asr
LD_LIBRARY_PATH=$PWD/build/cuda-asr/bin \
  ./build/cuda-asr/bin/nemo-speech transcribe audio.wav --model model.gguf
```

Pass `-DNEMO_SPEECH_CUBLAS_SHIM=OFF` to opt out and link real cuBLAS instead
(e.g. to A/B the shim against stock cuBLAS).

## Custom GPU kernels

The heavier project-specific CUDA kernels (fused rel-pos attention, skinny-Q8 GEMM,
NVFP4 quantization, BF16 FastConformer epilogues, fused LayerNorm, and F16
depthwise conv2d) live as ggml patches rather than in `kernels/` - see
[ggml patches](ggml-patches.md). `kernels/` holds only the cuBLAS shim and its
version map.
