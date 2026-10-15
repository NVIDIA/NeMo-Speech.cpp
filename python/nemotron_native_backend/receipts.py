from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import SynthesisRequest


def cohort_id(request_ids: Sequence[str], batch: int) -> str:
    if not request_ids:
        raise ValueError("cohort must contain requests")
    prefix = os.path.commonprefix(list(request_ids)).rstrip("-_/.")
    if prefix:
        return prefix
    return f"b{batch}-{time.time_ns()}"


class ReceiptWriter:
    def __init__(self, path: Path, *, manifest_sha256: str) -> None:
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("receipt path must be absolute with an existing parent")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("receipt path must be a regular file or absent, not a symlink")
        if any(component.is_symlink() for component in path.parents):
            raise ValueError("receipt path must not contain a symlink component")
        self.path = path
        self.manifest_sha256 = manifest_sha256
        self._lock = asyncio.Lock()

    async def write_success(
        self, requests: Sequence[SynthesisRequest], evidence: Mapping[str, Any]
    ) -> None:
        batch = len(requests)
        if batch not in (1, 16, 32, 64):
            raise ValueError("functional-v1 receipt requires an exact fleet batch")
        ids = [request.request_id for request in requests]
        payload = {
            "schema_version": 1,
            "cohort_id": cohort_id(ids, batch),
            "public_request_ids": ids,
            "real_weights": evidence["real_weights"],
            "execution_contract": evidence["execution_contract"],
            "actual_requests": batch,
            "predictor_plan_slices": list(evidence["predictor_plan_slices"]),
            "stage1_plan_slices": list(evidence["stage1_plan_slices"]),
            "decode_steps": evidence["decode_steps"],
            "predictor_plan_invocations": evidence["predictor_plan_invocations"],
            "predictor_passes_per_lane": evidence["predictor_passes_per_lane"],
            "padding_count": evidence["padding_count"],
            "predictor_fallback_count": evidence["predictor_fallback_count"],
            "stage1_fallback_count": evidence["stage1_fallback_count"],
            "code_h2d_copies": evidence["code_h2d_copies"],
            "code_d2h_copies": evidence["code_d2h_copies"],
            "manifest_sha256": self.manifest_sha256,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        async with self._lock:
            flags = (
                os.O_APPEND
                | os.O_CREAT
                | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self.path, flags, 0o640)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("receipt destination is not a regular file")
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError(f"short receipt write at byte {offset}")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
