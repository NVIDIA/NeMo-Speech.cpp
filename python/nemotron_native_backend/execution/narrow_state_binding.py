import copy
import ctypes as C
import hashlib
from pathlib import Path

from .engine import Stage0BuildInfo


def install(engine, narrow_library, narrow_sha256, artifact_root):
    path = Path(narrow_library)
    if hashlib.sha256(path.read_bytes()).hexdigest() != narrow_sha256:
        raise ValueError("BF16 narrow library changed")
    n = engine.core.neural
    probe = n.lib.mk2_b32_private_state_bytes
    probe.argtypes = []
    probe.restype = C.c_int
    if probe() != 2:
        raise ValueError("BF16 full producer format mismatch")
    initial = (C.c_void_p * 14)()
    ready = C.c_void_p()
    get = n.prefill_lib.mk2_r8_initial_state
    get.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    get.restype = C.c_int
    n._check(get(n.prefill_handle, initial, C.byref(ready)), "B32 BF16 canonical source")
    for i in range(14):
        private = int(n.bindings.ssm_state[i])
        if private <= int(initial[i]) < private + n.batch * 196608 * 2:
            raise ValueError("canonical state aliases BF16 private state")
    setter = n.lib.mk2_r8_bind_initial_state
    setter.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.c_int]
    setter.restype = C.c_int
    n._check(
        setter(n.stage0_handle, initial, ready, n.bindings.seq_len, n.inputs.prompt_positions),
        "B32 BF16 canonical binding",
    )
    n._r8_initial_state = {
        "initial": initial,
        "ready": ready,
        "bytes": 14 * 196608 * 4,
        "prefill_owner": n.prefill_handle,
    }
    engine._b32_prefill_state = {"canonical_bytes": 14 * 196608 * 4}
    old = engine.contraction
    if old is None or old.enabled:
        raise ValueError("fresh retained contraction owner required")
    config = copy.deepcopy(engine.active_lanes_config)
    config["stage0"] = {"path": str(path.resolve()), "sha256": narrow_sha256}
    owner_type = type(old)
    old.close()
    engine.contraction = owner_type(
        n,
        config,
        artifact_root,
        engine.residency["codec_blocks"],
        Stage0BuildInfo,
        engine.execution,
    )
    engine.active_lanes_config = config
    probe = engine.contraction.lib.mk2_b32_private_state_bytes
    probe.argtypes = []
    probe.restype = C.c_int
    if probe() != 2:
        raise ValueError("BF16 narrow producer format mismatch")
    engine._b32_prefill_state.update(
        variant="n1",
        private_state_bytes=14 * 32 * 196608 * 2,
        prefill_fp32_workspace_bytes=14 * 32 * 196608 * 4,
        narrow_library=str(path),
        narrow_sha256=narrow_sha256,
    )
