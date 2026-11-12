# Reviewing `patches/`

Each `.patch` file in `patches/` is a commit applied on top of the pinned `llama.cpp` submodule in the exact order listed in `patches/series`.

- Review the C++ and CUDA code inside each patch as compiled source code, not as plain text.
- Do not read patch hunks in isolation:
  1. Check the surrounding function and struct definitions in `llama.cpp/`. If the `llama.cpp/` checkout is empty, fetch `https://raw.githubusercontent.com/ggml-org/llama.cpp/<pin>/<file>`, where `<pin>` is the commit that the `llama.cpp` submodule points to (`git ls-tree HEAD llama.cpp` prints it). If that fails, use the context lines of the patch and the earlier patches that change the same function.
  2. Search other `patches/*.patch` files in `patches/series` that modify the same file, struct, or custom ggml/llama API.
  3. Check every caller in `src/`, `server/`, and `include/`. Include callers that use underlying streams, graphs, or buffers directly and bypass the normal wrapper entry points. Report a caller that uses a patched op or API outside a `NEMO_SPEECH_GGML_PATCHED` guard, because unpatched builds do not have it.
- Report an op whose `supports_op` or dispatch check accepts a case that the kernel does not handle, on discrete GPUs or on iGPU/UMA platforms.
- Audit unchanged context lines whose reachability or execution frequency changes (such as process-aborting checks becoming reachable on mid-run runtime failures or OOM). Anchor any finding on the changed line in the patch that altered reachability.
