from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class WorkerManifest(Protocol):
    sha256: str
    raw: dict[str, Any]
    enabled_buckets: tuple[int, ...]

    def worker_ready_fields(self) -> dict[str, Any]: ...
