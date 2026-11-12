# Review checklist

CodeRabbit and Greptile use this file as their review rules. Contributors and maintainers can use it too.

NeMo-Speech.cpp is a C++17 speech runtime (ASR, SpeechLM, diarization, TTS, translation, VoiceChat, text normalization) built on ggml from a pinned `llama.cpp` submodule, with a C SDK, CLI, HTTP/WebSocket and gRPC servers, and Python GGUF converters. It has CPU, CUDA, Metal and Vulkan support, with portable release archives for Linux, macOS and Windows across `x86_64` and `arm64`. Most custom kernels and the `llama.cpp` patch series target the CUDA backend. Supported hardware spans discrete GPUs, integrated/unified-memory (iGPU/UMA) platforms where CPU and GPU share memory (DGX Spark, RTX Spark, Jetson, Apple Silicon), and CPU-only hosts.

## What to report

- Priority: correctness, then performance, then portability, then maintainability.
- Report only problems that the change introduces or makes reachable.
- For each problem, name the input, configuration, backend or platform that reaches it, the concrete impact (wrong output, crash or abort, race, resource leak or waste, or latency regression), the file and line, and the fix. Do not report speculative issues without a reachable path.
- If a change makes unchanged surrounding or called code run more often or under new runtime conditions, audit that unchanged code too and attach the comment to the changed line that altered its reachability.
- Do not report formatting, naming, missing docstrings, or anything `.pre-commit-config.yaml` already checks.

## Look past the diff

Most missed bugs sit outside the changed hunks.

- Read the full function around each change, its callers, and the helpers it calls.
- Trace state across layers. If a change adds or updates state (counters, timestamps, caches, flags, buffers or backend handles), find every path that reads, writes, resets, copies or borrows it across `src/`, `server/`, `patches/` and `llama.cpp`. Include fallbacks, error paths, concurrent threads and requests, and callers that use underlying streams, graphs or buffers directly and bypass wrapper entry points.
- Check parallel and sibling paths for symmetry. When a change fixes a failure, moves a precondition check, or updates behavior in one place, inspect the parallel implementations. Examples: paired model stages, fast path and fallback, streaming and offline, HTTP/WebSocket and gRPC routes, `conversion/` and the C++ model loaders, Linux/macOS and Windows scripts, patched and unpatched ggml. Report siblings that keep the bug or run setup in a different order.

## Correctness and concurrency

- Look for wrong results, crashes, memory errors, races, integer/buffer overflows, and uninitialized or stale state.
- Look for edge cases: empty or very short inputs, boundary sizes, partial tiles/batches/chunks, masked or multi-speaker/multi-sequence state, and numerical instability.
- Fallback paths must produce results consistent with fast paths, and streaming results must not depend on chunk size unless documented.
- Multiple sessions, pipelines or backend instances can run concurrently in one process, or share one physical device across threads. Distinguish per-request, per-session, per-backend, device-wide and process-wide state. Update related shared fields atomically, under one lock or one atomic protocol.
- Validate preconditions before side effects. Checks that can reject a request, model or fast path (input/upload limits, shapes, quantization/data types, hardware capabilities) must run before allocations, full-input decoding, weight repacking/conversion, file writes, and global/device-wide settings. If they run later, the rejected path must undo every side effect.

## Failures at run time

- Process-terminating checks must not be reachable from recoverable runtime conditions. Such checks are `CUDA_CHECK`, `GGML_ASSERT`, `GGML_ABORT`, `abort`, and uncaught exceptions across C API or thread boundaries. Recoverable conditions include out of memory, resource exhaustion, failed runtime compilation or instantiation, bad client input, and missing files.
- Libraries and servers must fall back or return an error to the caller instead of terminating the process. On unified-memory (iGPU/UMA) systems, CPU and GPU share physical memory, so device allocations and runtime setup can fail mid-run long after startup succeeded.
- Clean up sticky error state on recovery paths. When a backend or OS call fails and the code recovers or falls back, clear thread-local sticky error state (for example `cudaGetLastError()`), so that a later check does not fail on the stale error.

## Performance

- On hot paths, look for added host-device or cross-thread synchronization, per-chunk or per-step allocations, redundant conversions or repacks, and extra host-device or cross-backend copies.
- Also look for silent CPU op fallbacks, caches that fail to hit or that invalidate all entries at once, and locks or stream waits that serialize work that ran in parallel before.

## Portability, packaging and ABI

- Look for assumptions tied to one GPU architecture, SM count, memory model (discrete vs. iGPU/UMA), CPU ISA (`x86_64` vs. `arm64`), backend, OS or compiler.
- Query hardware limits and architecture features at run time, guard architecture-specific instructions and kernels, and keep a working fallback.
- Keep `-DNEMO_SPEECH_GGML_PATCHED=OFF` builds compiling and working (`metal-*` and `vulkan-*` presets).
- Keep Python converters (`conversion/`, `convert_model.py`) consistent with the C++ GGUF loaders.
- Public headers in `include/nemo_speech/*.h` define a `dlopen`-compatible C ABI: do not reorder struct fields, change struct sizes unguarded, or break existing caller signatures.
- Code and scripts must work across GCC, Clang, Apple Clang and MSVC, and keep Linux, macOS and Windows release archives (`x86_64` and `arm64`) self-contained.

## Maintainability and reuse

Give a maintainability comment only when you can point to the code: the helper to reuse, the second copy, or the item that is not used.

- Reuse before you add. Search `src/common/` and `src/runtime/ggml/` for a helper that does the same work. Report new code that duplicates one, and name the helper.
- Report a block that the change copies with small edits. Ask for one shared function. This includes the same logic repeated across models, backends, and the HTTP and gRPC servers.
- Keep one source of truth. A constant, default, limit, tensor name or environment variable must have one definition that all code uses.
- Keep platform and backend branches in one place, behind one helper or interface. Report `#ifdef` or backend checks that the change spreads across call sites.
- Follow the patterns of the surrounding code for error handling, logging and resource cleanup. Report a new pattern only when an existing one does the same job.
- Remove what the change makes unused: dead code, unreachable branches, and unused parameters, flags and options. Do not leave commented-out code.
- Give each constant a name and a reason. Report a new threshold, size or timeout that has no name or no comment that says where the value comes from.
- Do not add options, parameters or abstraction layers that nothing uses yet.
- A bug fix or new behavior needs a test when a test can cover it. If the change adds none, name the test that is missing.
- Keep the change focused. Report unrelated refactors, reformatting or speculative code (see `CONTRIBUTING.md`).

## patches/

- Each `.patch` file in `patches/` modifies `llama.cpp` or `ggml` at the pinned commit of the `llama.cpp` submodule, applied in `patches/series` order. Review the patched code as compiled source code, not plain diff text.
- Inspect the full target function and struct in `llama.cpp/`, search all other `patches/*.patch` files in `patches/series` that touch the same file, struct or API, and check every caller in `src/`, `server/` and `include/`.
- Check that op `supports_op` / dispatch preconditions match what the kernels support, and that `patches/series` stays consistent.

## Docs and benchmarks

- Check that benchmark methodology and measurements are sound, and that documented numbers, environment variables, defaults, API payloads and CLI flags match the code.
- If a change alters a default, flag, API or runtime behavior, report documentation in `docs/`, `README.md` or `BENCHMARK.md` that no longer matches it.
