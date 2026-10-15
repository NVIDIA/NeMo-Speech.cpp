from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import AudioResult, SynthesisRequest


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    cuda_device_name: str
    cuda_compute_capability: tuple[int, int]
    enabled_buckets: tuple[int, ...]
    real_weights: bool
    execution_contract: str = "unspecified"


class SynthesisBackend(ABC):
    @property
    @abstractmethod
    def identity(self) -> BackendIdentity:
        raise NotImplementedError

    @abstractmethod
    async def synthesize_batch(self, requests: Sequence[SynthesisRequest]) -> list[AudioResult]:
        raise NotImplementedError

    def consume_execution_evidence(
        self, requests: Sequence[SynthesisRequest], results: Sequence[AudioResult]
    ) -> dict[str, Any]:
        raise BackendError("backend_evidence", "backend did not provide execution evidence")

    async def shutdown(self) -> None:
        return None


def exact16_execution_evidence(
    identity: BackendIdentity,
    requests: Sequence[SynthesisRequest],
    results: Sequence[AudioResult],
) -> dict[str, Any]:
    return {
        "real_weights": identity.real_weights,
        "execution_contract": identity.execution_contract,
        "request_ids": tuple(request.request_id for request in requests),
        "result_ids": tuple(result.request_id for result in results),
        "predictor_plan_slices": (len(requests),),
        "stage1_plan_slices": (len(requests),),
        "decode_steps": 15,
        "predictor_plan_invocations": 15,
        "predictor_passes_per_lane": 240,
        "padding_count": 0,
        "predictor_fallback_count": 0,
        "stage1_fallback_count": 0,
        "code_h2d_copies": 0,
        "code_d2h_copies": 0,
    }


def validate_exact16_execution_evidence(
    identity: BackendIdentity,
    requests: Sequence[SynthesisRequest],
    results: Sequence[AudioResult],
    evidence: Mapping[str, Any],
) -> None:
    expected = exact16_execution_evidence(identity, requests, results)
    if dict(evidence) != expected:
        raise BackendError("backend_evidence", "per-invocation exact16 evidence drift")


def backend_identity_dict(identity: BackendIdentity) -> dict[str, Any]:
    return {
        "cuda_device_name": identity.cuda_device_name,
        "cuda_compute_capability": list(identity.cuda_compute_capability),
        "enabled_buckets": list(identity.enabled_buckets),
        "real_weights": identity.real_weights,
        "execution_contract": identity.execution_contract,
    }
