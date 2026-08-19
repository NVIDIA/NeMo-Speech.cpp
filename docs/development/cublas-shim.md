# cuBLAS shim and GPU kernels

The minimal runtime image ships **no NVIDIA cuBLAS**. ggml-cuda's
non-quantized GEMMs (FastConformer attention, subsampling convs, CTC head; the
q8/RNNT weight matmuls already use ggml's quantized kernels) are served instead
by an in-tree drop-in `libcublas`.

## The shim

`kernels/cublas_shim.cu` is a drop-in cuBLAS library: shape-specialized CUDA
GEMM/GEMV kernels, including WMMA tensor-core paths, but **no cuBLASLt**.
Linux uses the generated symbol map from `kernels/ver_cublas.map`; Windows exports
the same ABI from `cublas64_<major>.dll`. The target inherits
`CMAKE_CUDA_ARCHITECTURES` when set and falls back to JIT-portable
`compute_80` PTX for ad-hoc builds. Dropping real cuBLAS and cuBLASLt is
the bulk of the package size. The shim is built separately from ggml.
Prebuilt releases include Turing (SM75) code. Local Turing builds must set
`CMAKE_CUDA_ARCHITECTURES=75` or `native`, because `compute_80` PTX cannot run
on SM75.

It's an optional CMake target, **`NEMO_SPEECH_CUBLAS_SHIM` (default `OFF`)**,
built when explicitly enabled with `GGML_CUDA` (a no-op for Metal, Vulkan,
and CPU builds). Normal shared-library source builds link the CUDA toolkit's
cuBLAS; static ggml builds may also link cuBLASLt. Portable container and
release-archive builds enable the shim and do not require those libraries at
run time. Linux uses a matching SONAME and symbol version; Windows uses the
matching versioned DLL name.

To build and exercise the container GEMM path outside the container, enable the
shim and put its output directory first on the loader path:

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
[ggml patches](ggml-patches.md). `kernels/` holds only the cuBLAS shim and its
version-map template.
