import ctypes as C
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parents[1]


def install(n, *, state_mode, scales_path, map_path, recipe_identity):
    if getattr(n, "_r9_state", None) is not None:
        return n._r9_state
    import torch

    from .startup_binding import install as startup_install

    mode = state_mode
    if n.started or n.batch not in (128, 256, 512, 1024, 2048, 4096):
        raise ValueError("fresh supported wide state owner required")
    initial = (C.c_void_p * 14)()
    ready = C.c_void_p()
    get = n.prefill_lib.mk2_r8_initial_state
    get.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    get.restype = C.c_int
    n._check(get(n.prefill_handle, initial, C.byref(ready)), "canonical state source")
    bind = n.lib.mk2_r8_bind_initial_state
    bind.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.c_int]
    bind.restype = C.c_int
    n._check(
        bind(n.stage0_handle, initial, ready, n.bindings.seq_len, n.inputs.prompt_positions),
        "canonical state bind",
    )
    recipe = recipe_identity
    scale_np = np.fromfile(scales_path, np.float32).reshape(14, 16384)
    map_np = np.fromfile(map_path, np.int32).reshape(14, 1536)
    scales = torch.tensor(scale_np, device="cuda")
    maps = torch.tensor(map_np, device="cuda")
    a = (C.c_void_p * 14)(*[scales[i].data_ptr() for i in range(14)])
    b = (C.c_void_p * 14)(*[maps[i].data_ptr() for i in range(14)])
    for lib, fn, handle in [
        (n.lib, "mk2_r9_state_bind", n.stage0_handle),
        (n.prefill_lib, "mk2_r9_prefill_bind", n.prefill_handle),
    ]:
        f = getattr(lib, fn)
        f.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        f.restype = C.c_int
        n._check(f(handle, a, b), fn)
    n._r8_initial_state = {
        "initial": initial,
        "ready": ready,
        "bytes": 14 * 196608 * 4,
        "prefill_owner": n.prefill_handle,
        "initial_positions": n.inputs.prompt_positions,
    }
    n._r9_state = {
        "mode": mode,
        "scales": scales,
        "maps": maps,
        "recipe": str(recipe),
        "state_bytes": 2 if mode == 1 else 1,
        "canonical": n._r8_initial_state,
        "private_bytes": 14 * n.batch * 196608 * (2 if mode == 1 else 1),
    }
    startup_install(n)
    return n._r9_state
