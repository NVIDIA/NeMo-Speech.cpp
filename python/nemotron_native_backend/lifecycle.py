from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "nemotron-speech.worker-lifecycle.v1"
ENVIRONMENT = "NEMOTRON_TRT_LIFECYCLE_RECEIPT"


class LifecycleRecorder:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._last_monotonic_ns = 0
        self.worker_exit: int | None = None
        if path is None:
            return
        if not path.is_absolute():
            raise RuntimeError("lifecycle receipt path must be absolute")
        if path.exists() or path.is_symlink():
            raise RuntimeError("lifecycle receipt must be absent at startup")
        if not path.parent.is_dir():
            raise RuntimeError("lifecycle receipt parent must exist")
        for parent in reversed(path.parents):
            if parent.is_symlink():
                raise RuntimeError("lifecycle receipt path has a symlink component")

    @classmethod
    def from_environment(cls, *, required: bool) -> LifecycleRecorder:
        value = os.environ.get(ENVIRONMENT)
        if value is None:
            if required:
                raise RuntimeError(f"{ENVIRONMENT} must be set")
            return cls(None)
        return cls(Path(value))

    @staticmethod
    def sentinel_stat(path: Path) -> dict[str, Any]:
        observed = os.lstat(path)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"stop sentinel must be a regular file: {path}")
        return {
            "path": str(path),
            "mode": stat.S_IMODE(observed.st_mode),
            "size": observed.st_size,
            "uid": observed.st_uid,
            "gid": observed.st_gid,
            "inode": observed.st_ino,
            "device": observed.st_dev,
            "mtime_ns": observed.st_mtime_ns,
        }

    def record(self, phase: str, **fields: Any) -> None:
        monotonic_ns = max(time.monotonic_ns(), self._last_monotonic_ns + 1)
        self._last_monotonic_ns = monotonic_ns
        event = {"phase": phase, "monotonic_ns": monotonic_ns, **fields}
        self.events.append(event)
        if phase == "worker_exit":
            self.worker_exit = int(fields["exit_code"])
        print(
            "nemotron_trt_lifecycle " + json.dumps(event, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )
        if self.path is not None:
            self._write()

    def _write(self) -> None:
        if self.path is None:
            raise RuntimeError("lifecycle recorder has no receipt path")
        payload = {
            "schema": SCHEMA,
            "status": "PASS" if self.worker_exit == 0 else "IN_PROGRESS",
            "pid": os.getpid(),
            "worker_exit": self.worker_exit,
            "forced": False,
            "events": self.events,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{len(self.events)}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, self.path)
            parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            if temporary.exists():
                temporary.unlink()


_RECORDER: LifecycleRecorder | None = None


def install(recorder: LifecycleRecorder) -> None:
    global _RECORDER
    _RECORDER = recorder


def record(phase: str, **fields: Any) -> None:
    if _RECORDER is not None:
        _RECORDER.record(phase, **fields)


__all__ = [
    "ENVIRONMENT",
    "LifecycleRecorder",
    "SCHEMA",
    "install",
    "record",
]
