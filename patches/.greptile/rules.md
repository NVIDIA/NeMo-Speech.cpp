# Reviewing `patches/`

Each `.patch` file in `patches/` is a commit applied on top of the pinned `llama.cpp` submodule in the exact order listed in `patches/series`.

- Review the C++ and CUDA code inside each patch as compiled source code, not as plain text.
- Do not read patch hunks in isolation:
  1. Check the surrounding function and struct definitions in `llama.cpp/`.
  2. Search other `patches/*.patch` files in `patches/series` that modify the same file, struct, or custom ggml/llama API.
  3. Check every caller in `src/`, `server/`, and `include/`. Include callers that use underlying streams, graphs, or buffers directly and bypass the normal wrapper entry points. Verify that `-DNEMO_SPEECH_GGML_PATCHED=OFF` builds still compile and work.
- Check op `supports_op` / dispatch preconditions against kernel limits, architecture guards, and fallback paths across discrete GPUs and iGPU/UMA platforms.
- Audit unchanged context lines whose reachability or execution frequency changes (such as process-aborting checks becoming reachable on mid-run runtime failures or OOM). Anchor any finding on the changed line in the patch that altered reachability.
