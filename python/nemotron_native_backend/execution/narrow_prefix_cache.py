import hashlib
import time


def install(engine):
    n = engine.core.neural
    t = engine.t
    initial = getattr(n, "_r8_initial_state", None)
    if initial is None or n.started or engine.batch != 32 or hasattr(engine, "_b32_prefix_owner"):
        raise ValueError("canonical prefix requires a fresh B32 canonical owner")
    private_bytes = engine._b32_prefill_state.get("private_state_bytes")
    if private_bytes not in (14 * 32 * 196608 * 2, 14 * 32 * 196608 * 4):
        raise ValueError("unrecognized private recurrent format")

    def view(pointer, count):
        class Alias:
            def __init__(self):
                self.__cuda_array_interface__ = {
                    "shape": (count,),
                    "typestr": "|u1",
                    "data": (int(pointer), False),
                    "version": 3,
                    "strides": None,
                }

        return t.as_tensor(Alias(), device="cuda:0")

    b = n.batch
    p = n.inputs.prompt_positions
    blocks = n.info.blocks_per_sequence
    original = n.prefill_lib.mk2_native_prefill_run
    version = n.inputs.prompt._version
    context = tuple(tuple(request.context_ids) for request in n.inputs.requests)
    identity = (
        n.prefill_handle,
        n.stage0_handle,
        tuple(int(initial["initial"][i]) for i in range(14)),
        initial["ready"].value,
        tuple(int(n.bindings.ssm_state[i]) for i in range(14)),
    )
    views = [n.final_hidden, view(n.bindings.slot_map, b * 4), view(n.bindings.seq_len, b * 4)]
    for layer in range(14):
        views.append(view(n.bindings.conv_state[layer], b * n.bindings.conv_slot_stride[layer] * 2))
        views.append(view(initial["initial"][layer], 196608 * 4))
    for layer in range(3):
        views.append(view(n.bindings.kv_block_table[layer], b * 2 * blocks * 4))
        pool = view(n.bindings.kv_pool_base[layer], b * 2 * blocks * 4 * 32 * 128 * 2).reshape(
            b, 2, blocks, 4, 32, 256
        )
        for block in range((p + 31) // 32):
            views.append(pool[:, :, block, :, : min(32, p - block * 32), :])
    ready = view(initial["ready"].value, 4)
    views.append(ready)
    started = time.monotonic()
    with t.cuda.stream(n.stream):
        n._check(
            original(n.prefill_handle, n.stream.cuda_stream),
            "canonical prefix preparation",
            n.prefill_lib,
        )
        snapshots = [value.clone() for value in views]
    n.stream.synchronize()
    if ready.view(t.int32).item() != 1:
        raise ValueError("prefix did not establish a canonical initial state")
    receipt = {
        "kind": "canonical-FP32-prefix",
        "private_recurrent_storage_restored": False,
        "bytes": sum(value.numel() * value.element_size() for value in snapshots),
        "preparation_seconds": time.monotonic() - started,
        "hits": 0,
        "misses": 0,
        "prefill_sha256": n.prefill_library_sha256,
        "stage0_sha256": n.library_sha256,
        "prompt_sha256": hashlib.sha256(
            n.inputs.prompt.cpu().view(t.uint8).numpy().tobytes()
        ).hexdigest(),
    }

    def run(handle, stream):
        current = (
            n.prefill_handle,
            n.stage0_handle,
            tuple(int(initial["initial"][i]) for i in range(14)),
            initial["ready"].value,
            tuple(int(n.bindings.ssm_state[i]) for i in range(14)),
        )
        if handle != n.prefill_handle or stream != n.stream.cuda_stream or current != identity:
            raise RuntimeError("canonical prefix owner or stream changed")
        if (
            n.inputs.prompt._version != version
            or tuple(tuple(request.context_ids) for request in n.inputs.requests) != context
        ):
            receipt["misses"] += 1
            return original(handle, stream)
        with t.cuda.stream(n.stream):
            for destination, snapshot in zip(views, snapshots, strict=True):
                destination.copy_(snapshot)
        receipt["hits"] += 1
        return 0

    n.prefill_lib.mk2_native_prefill_run = run
    engine._b32_prefix_owner = (views, snapshots, original, run)
    engine._b32_canonical_cache = receipt

    engine._b32_canonical_cache["ownership_contract"] = (
        "exclusive fixed-allocation engine ownership; supported prompt/context reset "
        "invalidates; checkpoint, precision and state-format changes require a fresh "
        "engine/cache"
    )
