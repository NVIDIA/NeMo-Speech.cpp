from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import (
    SAMPLE_RATE_HZ,
    TEXT_EOS_TOKEN_ID,
    RequestError,
    _strict_submit_fields,
)
from .streaming_contract import STREAMING_MAX_TOKENS

STREAMING_OUTPUT_CONTRACT = "pcm_f32_le_22050_mono_variable_streaming_v2"
STREAMING_FLEET_EXECUTION_CONTRACT = (
    "trtllm_stage0_natural512_trt_predictor_fleet_gpu_bridge_trt_stage1_stateful_fleet_v2"
)


@dataclass(frozen=True, slots=True)
class StreamingSynthesisRequest:
    request_id: str
    session_id: str | None
    normalized_text: str
    token_ids: tuple[int, ...]
    speaker_id: str
    max_tokens: int
    temperature: float
    top_k: int
    sample_rate_hz: int
    seed: int

    @classmethod
    def from_message(cls, value: dict[str, Any]) -> StreamingSynthesisRequest:
        request = cls(**_strict_submit_fields(value))
        request.validate()
        return request

    def validate(self) -> None:
        if not self.request_id or len(self.request_id.encode("utf-8")) > 1024:
            raise RequestError("request_id must contain 1..1024 UTF-8 bytes")
        if not self.normalized_text.strip():
            raise RequestError("normalized_text must not be empty")
        if len(self.normalized_text.encode("utf-8")) > 1_000_000:
            raise RequestError("normalized_text is too large")
        if len(self.token_ids) > 4096:
            raise RequestError("token_ids exceeds 4096")
        if any(item < 0 or item >= 131_075 for item in self.token_ids):
            raise RequestError("token_ids contains an out-of-vocabulary ID")
        if self.token_ids[-1] != TEXT_EOS_TOKEN_ID:
            raise RequestError("streaming v2 requires the configured text EOS token")
        if self.speaker_id != "english_sample":
            raise RequestError("streaming v2 supports only speaker_id=english_sample")
        if self.max_tokens != STREAMING_MAX_TOKENS:
            raise RequestError("streaming v2 requires max_tokens=512")
        if not math.isfinite(self.temperature) or self.temperature != 0.7:
            raise RequestError("streaming v2 requires temperature=0.7")
        if self.top_k != 80:
            raise RequestError("streaming v2 requires top_k=80")
        if self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise RequestError("streaming v2 requires sample_rate_hz=22050")
        if self.seed < 0 or self.seed > 0xFFFFFFFFFFFFFFFF:
            raise RequestError("seed must fit uint64")

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.speaker_id,
            self.max_tokens,
            self.temperature,
            self.top_k,
            self.sample_rate_hz,
        )


@dataclass(frozen=True, slots=True)
class StreamingAudioChunk:
    request_id: str
    sequence: int
    pcm_f32_le: bytes
    samples: int
    is_final: bool
    sample_rate_hz: int = SAMPLE_RATE_HZ
    channels: int = 1

    def validate(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise RuntimeError("streaming audio chunk request_id is empty or not a string")
        if type(self.sequence) is not int or self.sequence < 0:
            raise RuntimeError("streaming audio chunk sequence is negative")

        if type(self.samples) is not int or self.samples < 0:
            raise RuntimeError("streaming audio chunk sample count is negative")
        if self.samples == 0 and not self.is_final:
            raise RuntimeError("streaming audio chunk must be nonempty")
        if type(self.is_final) is not bool:
            raise RuntimeError("streaming audio chunk final flag is not a boolean")
        if type(self.sample_rate_hz) is not int or type(self.channels) is not int:
            raise RuntimeError("streaming audio chunk rate and channels must be integers")
        if self.sample_rate_hz != SAMPLE_RATE_HZ or self.channels != 1:
            raise RuntimeError("streaming audio chunk is not 22050 Hz mono")
        if type(self.pcm_f32_le) is not bytes:
            raise RuntimeError("streaming PCM-F32LE payload must be bytes")
        if len(self.pcm_f32_le) != self.samples * 4:
            raise RuntimeError("streaming PCM-F32LE byte count differs from samples")


@dataclass(frozen=True, slots=True)
class StreamingResult:
    request_id: str
    chunk_count: int
    total_samples: int

    def validate(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise RuntimeError("streaming result request_id is empty or not a string")
        if type(self.chunk_count) is not int or type(self.total_samples) is not int:
            raise RuntimeError("streaming result counts must be integers")
        if self.chunk_count <= 0 or self.total_samples <= 0:
            raise RuntimeError("streaming result must contain nonempty audio")


__all__ = [
    "STREAMING_OUTPUT_CONTRACT",
    "STREAMING_FLEET_EXECUTION_CONTRACT",
    "StreamingAudioChunk",
    "StreamingResult",
    "StreamingSynthesisRequest",
]
