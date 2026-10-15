from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE_HZ = 22_050
SAMPLES_PER_REQUEST = 17_640
PCM_BYTES_PER_REQUEST = SAMPLES_PER_REQUEST * 4
TEXT_EOS_TOKEN_ID = 131_073


class RequestError(ValueError):
    pass


_SUBMIT_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "request_id",
        "session_id",
        "normalized_text",
        "token_ids",
        "speaker_id",
        "max_tokens",
        "temperature",
        "top_k",
        "sample_rate_hz",
        "seed",
    }
)
_STRING_SUBMIT_FIELDS = ("request_id", "normalized_text", "speaker_id")
_INTEGER_SUBMIT_FIELDS = ("max_tokens", "top_k", "sample_rate_hz", "seed")


def _strict_submit_fields(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RequestError("submit payload must be an object with string keys")
    observed = set(value)
    if observed != _SUBMIT_FIELDS:
        raise RequestError(
            f"submit fields differ: missing={sorted(_SUBMIT_FIELDS - observed)}, "
            f"extra={sorted(observed - _SUBMIT_FIELDS)}"
        )
    if type(value["protocol_version"]) is not int or value["protocol_version"] != 1:
        raise RequestError("submit protocol_version must be integer 1")
    if type(value["kind"]) is not str or value["kind"] != "submit":
        raise RequestError("submit kind must be the string submit")
    for name in _STRING_SUBMIT_FIELDS:
        if type(value[name]) is not str:
            raise RequestError(f"{name} must be a string")
    session_id = value["session_id"]
    if session_id is not None and type(session_id) is not str:
        raise RequestError("session_id must be a string or null")
    if session_id is not None and (not session_id or len(session_id.encode("utf-8")) > 1024):
        raise RequestError("session_id must contain 1..1024 UTF-8 bytes when present")
    token_ids = value["token_ids"]
    if not isinstance(token_ids, list) or not token_ids:
        raise RequestError("token_ids must be a nonempty array")
    if any(type(item) is not int for item in token_ids):
        raise RequestError("token_ids values must be integers")
    for name in _INTEGER_SUBMIT_FIELDS:
        if type(value[name]) is not int:
            raise RequestError(f"{name} must be an integer")
    if type(value["temperature"]) is not float:
        raise RequestError("temperature must be a float")
    return {
        "request_id": value["request_id"],
        "session_id": session_id,
        "normalized_text": value["normalized_text"],
        "token_ids": tuple(token_ids),
        "speaker_id": value["speaker_id"],
        "max_tokens": value["max_tokens"],
        "temperature": value["temperature"],
        "top_k": value["top_k"],
        "sample_rate_hz": value["sample_rate_hz"],
        "seed": value["seed"],
    }


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
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
    def from_message(cls, value: dict[str, Any]) -> SynthesisRequest:
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
            raise RequestError("functional v1 requires the configured text EOS token")
        if self.speaker_id != "english_sample":
            raise RequestError("functional v1 supports only speaker_id=english_sample")
        if self.max_tokens != 16:
            raise RequestError("functional v1 requires max_tokens=16 (D15 plus sentinel)")
        if not math.isfinite(self.temperature) or self.temperature != 0.7:
            raise RequestError("functional v1 requires temperature=0.7")
        if self.top_k != 80:
            raise RequestError("functional v1 requires top_k=80")
        if self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise RequestError("functional v1 requires sample_rate_hz=22050")
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
class AudioResult:
    request_id: str
    pcm_f32_le: bytes
    samples: int = SAMPLES_PER_REQUEST
    sample_rate_hz: int = SAMPLE_RATE_HZ
    channels: int = 1

    def validate(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise RuntimeError("backend output request_id is empty or not a string")
        if type(self.samples) is not int:
            raise RuntimeError("backend output sample count is not an integer")
        if self.samples != SAMPLES_PER_REQUEST:
            raise RuntimeError("backend output sample count is not 17640")
        if type(self.sample_rate_hz) is not int or type(self.channels) is not int:
            raise RuntimeError("backend output rate and channels must be integers")
        if self.sample_rate_hz != SAMPLE_RATE_HZ or self.channels != 1:
            raise RuntimeError("backend output is not 22050 Hz mono")
        if type(self.pcm_f32_le) is not bytes:
            raise RuntimeError("backend PCM-F32LE payload must be bytes")
        if len(self.pcm_f32_le) != PCM_BYTES_PER_REQUEST:
            raise RuntimeError(
                f"backend PCM-F32LE byte count {len(self.pcm_f32_le)} != {PCM_BYTES_PER_REQUEST}"
            )
