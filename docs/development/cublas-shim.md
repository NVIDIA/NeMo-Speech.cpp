# cuBLAS shim and GPU kernels

The minimal runtime image ships **no NVIDIA cuBLAS**. ggml-cuda's
non-quantized GEMMs (FastConformer attention, subsampling convs, CTC head; the
q8/RNNT weight matmuls already use ggml's quantized kernels) are served instead
by an in-tree drop-in `libcublas`.

## The shim

`kernels/cublas_shim.cu` (with the generated symbol map from
`kernels/ver_cublas.map`) is a drop-in cuBLAS library: shape-specialized CUDA
GEMM/GEMV kernels, including WMMA tensor-core paths, but **no cuBLASLt**. It inherits
`CMAKE_CUDA_ARCHITECTURES` when set and falls back to JIT-portable `compute_80`
PTX for ad-hoc builds. Dropping real cuBLAS + cuBLASLt is the bulk of the
container size. The shim is built as a separate library from ggml.

It's an optional CMake target, **`NEMO_SPEECH_CUBLAS_SHIM` (default `OFF`)**,
built when explicitly enabled together with `GGML_CUDA` (Linux only,
auto-skipped on Windows; a no-op for Metal, Vulkan, and CPU builds). Normal
source builds therefore link the CUDA toolkit's cuBLAS and cuBLASLt. Container
and release-archive builds enable the shim explicitly and skip those libraries
in their runtime closure. The generated SONAME and symbol version match the
CUDA toolkit major used for the build.

To build and exercise the container GEMM path outside the container, enable the
shim and put its output directory first on the loader path:

```bash
scripts/configure.sh cuda-asr -DNEMO_SPEECH_CUBLAS_SHIM=ON
cmake --build --preset cuda-asr
LD_LIBRARY_PATH=$PWD/build/cuda-asr/bin \
  ./build/cuda-asr/bin/nemo-speech transcribe audio.wav --model model.gguf
```

## Custom GPU kernels

The heavier project-specific CUDA kernels (fused rel-pos attention, skinny-Q8 GEMM,
NVFP4 quantization, BF16 FastConformer epilogues, fused LayerNorm, and F16
depthwise conv2d) live as ggml patches rather than in `kernels/` - see
[ggml patches](ggml-patches.md). `kernels/` holds only the cuBLAS shim and its
version-map template.
