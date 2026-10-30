# cuBLAS shim and GPU kernels

The release archives and the container ship NVIDIA's cuBLAS and cuBLASLt. For a
smaller runtime closure, a build can serve ggml-cuda's non-quantized GEMMs
(FastConformer attention, subsampling convs, CTC head; the q8/RNNT weight
matmuls already use ggml's quantized kernels) from an in-tree drop-in
`libcublas` instead.

## The shim

`kernels/cublas_shim.cu` is a drop-in cuBLAS library: shape-specialized CUDA
GEMM/GEMV kernels, including WMMA tensor-core paths, but **no cuBLASLt**.
Linux uses the generated symbol map from `kernels/ver_cublas.map`; Windows exports
the same ABI from `cublas64_<major>.dll`. The target inherits
`CMAKE_CUDA_ARCHITECTURES` when set and falls back to JIT-portable
`compute_80` PTX for ad-hoc builds. Real cuBLAS and cuBLASLt are the bulk of
a CUDA package's size. The shim is built separately from ggml. Turing builds
must set `CMAKE_CUDA_ARCHITECTURES=75` or `native`, because `compute_80` PTX
cannot run on SM75.

It's an optional CMake target, **`NEMO_SPEECH_CUBLAS_SHIM` (default `OFF`)**,
built when explicitly enabled with `GGML_CUDA` (a no-op for Metal, Vulkan,
and CPU builds). Normal shared-library source builds link the CUDA toolkit's
cuBLAS; static ggml builds may also link cuBLASLt. A build with the shim does
not require those libraries at run time; the release archives and the
container do not use it. Linux uses a matching SONAME and symbol version;
Windows uses the matching versioned DLL name.

To build and exercise the shim's GEMM path, enable it and put its output
directory first on the loader path:

```bash
scripts/configure.sh cuda-asr \
  -DNEMO_SPEECH_CUBLAS_SHIM=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset cuda-asr
LD_LIBRARY_PATH=$PWD/build/cuda-asr/bin \
  ./build/cuda-asr/bin/nemo-speech transcribe audio.wav --model model.gguf
```

On Windows:

```powershell
.\scripts\windows\build.ps1 -Backend cuda -CublasShim
```

## Custom GPU kernels

The heavier project-specific CUDA kernels (fused rel-pos attention, skinny-Q8 GEMM,
NVFP4 quantization, BF16 FastConformer epilogues, fused LayerNorm, and F16
depthwise conv2d) live as ggml patches rather than in `kernels/` - see
[`patches/`](../../patches/README.md). `kernels/` holds only the cuBLAS shim and its
version-map template.
