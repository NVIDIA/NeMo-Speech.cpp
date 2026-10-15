from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InflightPolicy:
    codec_merge_ready: bool = False
    codec_deadline_priority: bool = False
    codec_first_packet_f2: bool = False
    codec_local_packing: bool = False
    codec_packed_transfer: bool = False
    serial_prefetch: bool = False
    partial_startup_single_step: bool = False
    startup_phase_alignment: bool = False
    retire_batch: bool = False
    defer_admission_reset: bool = False
    producer_priority: str = "startup"
    startup_max_steps: int = 7
    startup_budget_ms: float = 20.0
    diagnostics: bool = False
    playback_reserve_guard: bool = False

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unsupported in-flight policy fields")
        policy = cls(**value)
        if any(
            type(getattr(policy, name)) is not bool
            for name in (
                "codec_merge_ready",
                "codec_deadline_priority",
                "codec_first_packet_f2",
                "codec_local_packing",
                "codec_packed_transfer",
                "serial_prefetch",
                "partial_startup_single_step",
                "startup_phase_alignment",
                "retire_batch",
                "defer_admission_reset",
                "diagnostics",
                "playback_reserve_guard",
            )
        ):
            raise ValueError("in-flight policy switches must be boolean")
        if policy.codec_deadline_priority and policy.codec_merge_ready:
            raise ValueError("codec deadline selection requires separate frame classes")
        if policy.codec_packed_transfer and not policy.codec_local_packing:
            raise ValueError("packed codec transfer requires codec local packing")
        if policy.producer_priority not in ("startup", "shared"):
            raise ValueError("unsupported producer priority")
        if type(policy.startup_max_steps) is not int or not 1 <= policy.startup_max_steps <= 7:
            raise ValueError("startup priority must be bounded to one through seven steps")
        if (
            type(policy.startup_budget_ms) not in (int, float)
            or not 0 < policy.startup_budget_ms <= 20
        ):
            raise ValueError("startup priority must be bounded to twenty milliseconds")
        return policy

    def record(self):
        return asdict(self)


DEFAULT_POLICY = InflightPolicy()
