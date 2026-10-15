from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

AUDIO_CODEBOOKS = 16
AUDIO_CODEC_TOKEN_MAX = 1023
AUDIO_MODEL_TOKEN_MAX = 1031
AUDIO_SENTINEL = -1
STREAMING_MAX_TOKENS = 512
MAX_GENERATED_RAW_ROWS = STREAMING_MAX_TOKENS - 1
STREAMING_SPEECH_DELAY = 5
SPEECH_DELAY_ENV = "NEMOTRON_STREAMING_SPEECH_DELAY"


def resolved_speech_delay() -> int:
    raw = os.environ.get(SPEECH_DELAY_ENV, "")
    if raw == "":
        return STREAMING_SPEECH_DELAY
    if raw not in {"3", "4", "5"}:
        raise StreamingContractError(f"{SPEECH_DELAY_ENV} must be 3, 4, or 5; observed {raw!r}")
    return int(raw)


MR2_MATCHED_STARTUP_CHUNKS = (6, 6)
MR2_MATCHED_STEADY_CHUNK = 8
OPTIMIZED_H100_STARTUP_CHUNKS = (10, 14, 16)
OPTIMIZED_H100_STEADY_CHUNK = 12
LOW_TTFA_H100_STARTUP_CHUNKS = (3, 5)
LOW_TTFA_H100_F1_STARTUP_CHUNKS = (1, 5)


LOW_TTFA_H100_R4_STARTUP_CHUNKS = (1, 2, 2, 3)


LOW_TTFA_H100_STEADY_CHUNK = 8
STREAMING_TAP_CAPACITY = 512
SUPPORTED_EXACT_BATCHES = (1, 16, 32, 64)
ZERO_CODEC_ROW = (0,) * AUDIO_CODEBOOKS


class StreamingContractError(RuntimeError):
    pass


def _row(value: Any, *, generated: bool) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StreamingContractError("audio-code row is not a sequence")
    result = tuple(int(item) for item in value)
    if len(result) != AUDIO_CODEBOOKS:
        raise StreamingContractError(
            f"audio-code row has {len(result)} values, expected {AUDIO_CODEBOOKS}"
        )
    if generated:
        if any(item < 0 or item > AUDIO_MODEL_TOKEN_MAX for item in result):
            raise StreamingContractError(
                f"generated audio codes must be in [0,{AUDIO_MODEL_TOKEN_MAX}]"
            )
    elif result != (AUDIO_SENTINEL,) * AUDIO_CODEBOOKS:
        raise StreamingContractError(
            "first audio-code row is not the exact [-1]*16 context sentinel"
        )
    return result


def classify_generated_row(value: Any) -> tuple[tuple[int, ...], str]:
    row = _row(value, generated=True)
    if not any(row):
        return row, "zero"
    if max(row) > AUDIO_CODEC_TOKEN_MAX:
        return row, "control"
    return row, "codec"


@dataclass(frozen=True, slots=True)
class ObservedAudioRow:
    generated_index: int
    codes: tuple[int, ...]
    classification: str


class CumulativeAudioCodeCursor:
    def __init__(self, *, speech_delay: int = STREAMING_SPEECH_DELAY) -> None:
        self.speech_delay = int(speech_delay)
        if self.speech_delay < 0:
            raise ValueError("speech_delay must be nonnegative")
        self._rows: tuple[tuple[int, ...], ...] = ()

    @property
    def observed_rows(self) -> int:
        return len(self._rows)

    @property
    def generated_rows(self) -> int:
        return max(0, len(self._rows) - 1)

    def consume(
        self, rows: Sequence[Sequence[int]]
    ) -> tuple[tuple[ObservedAudioRow, ...], tuple[tuple[int, ...], ...]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise StreamingContractError("audio-code snapshot is not a sequence")
        if not rows:
            raise StreamingContractError("audio-code snapshot is empty")
        normalized = (_row(rows[0], generated=False),) + tuple(
            _row(value, generated=True) for value in rows[1:]
        )
        if len(normalized) < len(self._rows):
            raise StreamingContractError(
                f"cumulative audio-code count regressed from {len(self._rows)} to {len(normalized)}"
            )
        if normalized[: len(self._rows)] != self._rows:
            raise StreamingContractError(
                "cumulative audio codes changed an already observed prefix"
            )
        first = max(len(self._rows), 1)
        self._rows = normalized
        observed: list[ObservedAudioRow] = []
        codec: list[tuple[int, ...]] = []
        for row_index in range(first, len(normalized)):
            codes, classification = classify_generated_row(normalized[row_index])
            generated_index = row_index - 1
            observed.append(ObservedAudioRow(generated_index, codes, classification))
            if classification == "codec" and generated_index >= self.speech_delay:
                codec.append(codes)
        return tuple(observed), tuple(codec)

    def finish(self) -> int:
        if not self._rows:
            raise StreamingContractError("stream ended without the context sentinel")
        if len(self._rows) == 1:
            raise StreamingContractError("stream ended without a generated audio-code row")
        return len(self._rows) - 1


@dataclass(frozen=True, slots=True)
class ChunkSchedule:
    profile: str
    startup: tuple[int, ...]
    steady: int

    def validate(self) -> None:
        if not self.profile or not self.startup:
            raise ValueError("chunk schedule profile/startup must be nonempty")
        if any(not 1 <= value <= 16 for value in (*self.startup, self.steady)):
            raise ValueError("chunk schedule frames must be in [1,16]")


MR2_MATCHED_CHUNK_SCHEDULE = ChunkSchedule(
    "mr2_matched", MR2_MATCHED_STARTUP_CHUNKS, MR2_MATCHED_STEADY_CHUNK
)
OPTIMIZED_H100_CHUNK_SCHEDULE = ChunkSchedule(
    "optimized_h100",
    OPTIMIZED_H100_STARTUP_CHUNKS,
    OPTIMIZED_H100_STEADY_CHUNK,
)
LOW_TTFA_H100_CHUNK_SCHEDULE = ChunkSchedule(
    "low_ttfa_h100",
    LOW_TTFA_H100_STARTUP_CHUNKS,
    LOW_TTFA_H100_STEADY_CHUNK,
)
LOW_TTFA_H100_F1_CHUNK_SCHEDULE = ChunkSchedule(
    "low_ttfa_h100_f1",
    LOW_TTFA_H100_F1_STARTUP_CHUNKS,
    LOW_TTFA_H100_STEADY_CHUNK,
)
LOW_TTFA_H100_R4_CHUNK_SCHEDULE = ChunkSchedule(
    "low_ttfa_h100_r4",
    LOW_TTFA_H100_R4_STARTUP_CHUNKS,
    LOW_TTFA_H100_STEADY_CHUNK,
)
CHUNK_SCHEDULES = {
    value.profile: value
    for value in (
        MR2_MATCHED_CHUNK_SCHEDULE,
        OPTIMIZED_H100_CHUNK_SCHEDULE,
        LOW_TTFA_H100_CHUNK_SCHEDULE,
        LOW_TTFA_H100_F1_CHUNK_SCHEDULE,
        LOW_TTFA_H100_R4_CHUNK_SCHEDULE,
    )
}


SERVICE_TO_STAGE1_CHUNK_PROFILE = {
    "mr2_matched": "mr2_matched",
    "optimized_h100": "optimized",
    "low_ttfa_h100": "low_ttfa",
    "low_ttfa_h100_f1": "low_ttfa_f1",
    "low_ttfa_h100_r4": "low_ttfa_r4",
}


def target_chunk_frames(
    chunk_index: int, schedule: ChunkSchedule = MR2_MATCHED_CHUNK_SCHEDULE
) -> int:
    value = int(chunk_index)
    if value < 0:
        raise ValueError("chunk index must be nonnegative")
    schedule.validate()
    if value < len(schedule.startup):
        return schedule.startup[value]
    return schedule.steady


@dataclass(frozen=True, slots=True)
class CohortCodeChunk:
    chunk_index: int
    model_frames: int
    codes: tuple[tuple[tuple[int, ...], ...], ...]
    true_lengths: tuple[int, ...]
    final_by_lane: tuple[bool, ...]


class ExactCohortChunker:
    def __init__(
        self,
        request_ids: Sequence[str],
        *,
        schedule: ChunkSchedule = MR2_MATCHED_CHUNK_SCHEDULE,
    ) -> None:
        self.request_ids = tuple(str(value) for value in request_ids)
        if len(self.request_ids) not in SUPPORTED_EXACT_BATCHES:
            raise ValueError(f"unsupported exact streaming cohort B{len(self.request_ids)}")
        if any(not value for value in self.request_ids):
            raise ValueError("request IDs must be nonempty")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request IDs must be unique")
        self._lane = {value: index for index, value in enumerate(self.request_ids)}
        schedule.validate()
        self.schedule = schedule
        self._pending: list[list[tuple[int, ...]]] = [[] for _ in self.request_ids]
        self._ended = [False for _ in self.request_ids]
        self._final_reported = [False for _ in self.request_ids]
        self._chunk_index = 0

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    @property
    def complete(self) -> bool:
        return all(self._ended) and not any(self._pending) and all(self._final_reported)

    def take_terminal_lanes(self) -> tuple[int, ...]:
        result: list[int] = []
        for lane, (ended, pending, reported) in enumerate(
            zip(self._ended, self._pending, self._final_reported, strict=True)
        ):
            if ended and not pending and not reported:
                self._final_reported[lane] = True
                result.append(lane)
        return tuple(result)

    def push(
        self,
        request_id: str,
        frames: Sequence[Sequence[int]] = (),
        *,
        ended: bool = False,
    ) -> None:
        try:
            lane = self._lane[request_id]
        except KeyError as error:
            raise StreamingContractError(
                f"request {request_id!r} is not in this exact cohort"
            ) from error
        if self._ended[lane]:
            raise StreamingContractError(
                f"request {request_id!r} received data after terminal state"
            )
        for value in frames:
            row, classification = classify_generated_row(value)
            if classification != "codec":
                raise StreamingContractError(
                    "cohort chunker accepts only already-filtered codec rows"
                )
            self._pending[lane].append(row)
        if ended:
            self._ended[lane] = True

    def pop_ready(self) -> CohortCodeChunk | None:
        if self.complete:
            return None
        target = target_chunk_frames(self._chunk_index, self.schedule)
        if any(
            len(pending) < target and not ended
            for pending, ended in zip(self._pending, self._ended, strict=True)
        ):
            return None
        all_short_terminal = all(self._ended) and all(
            len(pending) < target for pending in self._pending
        )
        model_frames = (
            max((len(pending) for pending in self._pending), default=0)
            if all_short_terminal
            else target
        )
        if model_frames <= 0:
            return None
        codes: list[tuple[tuple[int, ...], ...]] = []
        true_lengths: list[int] = []
        final_by_lane: list[bool] = []
        for lane, (pending, ended) in enumerate(zip(self._pending, self._ended, strict=True)):
            true_length = min(len(pending), model_frames)
            prefix = tuple(pending[:true_length])
            del pending[:true_length]
            codes.append(prefix + (ZERO_CODEC_ROW,) * (model_frames - true_length))
            true_lengths.append(true_length)
            final = ended and not pending and not self._final_reported[lane]
            final_by_lane.append(final)
            if final:
                self._final_reported[lane] = True
        result = CohortCodeChunk(
            chunk_index=self._chunk_index,
            model_frames=model_frames,
            codes=tuple(codes),
            true_lengths=tuple(true_lengths),
            final_by_lane=tuple(final_by_lane),
        )
        self._chunk_index += 1
        return result


@dataclass(frozen=True, slots=True)
class PublishedAudioChunk:
    request_id: str
    sequence: int
    payload: Any
    samples: int
    is_final: bool


class OneChunkLookaheadPublisher:
    def __init__(
        self, request_ids: Sequence[str], *, eager: bool = False, completion_marker: bool = False
    ) -> None:
        self.request_ids = tuple(str(value) for value in request_ids)
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("publisher request IDs must be nonempty and unique")
        self._lane = {value: index for index, value in enumerate(self.request_ids)}
        self._pending: list[tuple[Any, int] | None] = [None] * len(self.request_ids)
        self._sequence = [0] * len(self.request_ids)
        self._terminal = [False] * len(self.request_ids)
        self.eager = bool(eager)
        self.completion_marker = bool(completion_marker)
        if self.completion_marker and not self.eager:
            raise ValueError("completion markers exist only for eager publication")

        self._published_any = [False] * len(self.request_ids)

    def _index(self, request_id: str) -> int:
        try:
            return self._lane[request_id]
        except KeyError as error:
            raise StreamingContractError(
                f"request {request_id!r} is not in this publisher"
            ) from error

    def _publish(self, lane: int, *, final: bool) -> PublishedAudioChunk:
        pending = self._pending[lane]
        if pending is None:
            raise StreamingContractError("stream ended without a nonempty decoded audio chunk")
        payload, samples = pending
        self._pending[lane] = None
        result = PublishedAudioChunk(
            request_id=self.request_ids[lane],
            sequence=self._sequence[lane],
            payload=payload,
            samples=samples,
            is_final=final,
        )
        self._sequence[lane] += 1
        self._published_any[lane] = True
        if final:
            self._terminal[lane] = True
        return result

    def decoded(
        self,
        request_id: str,
        payload: Any,
        samples: int,
        *,
        terminal: bool = False,
        continuation_expected: bool = True,
    ) -> tuple[PublishedAudioChunk, ...]:
        lane = self._index(request_id)
        if self._terminal[lane]:
            raise StreamingContractError("decoded audio arrived after final publication")
        count = int(samples)
        if count <= 0:
            raise StreamingContractError("decoded audio chunk must contain samples")
        emitted: list[PublishedAudioChunk] = []
        if self._pending[lane] is not None:
            emitted.append(self._publish(lane, final=False))
        self._pending[lane] = (payload, count)
        if terminal:
            emitted.append(self._publish(lane, final=True))
        elif self.eager and continuation_expected:
            emitted.append(self._publish(lane, final=False))
        return tuple(emitted)

    def observe_codec_continuation(self, request_id: str) -> tuple[PublishedAudioChunk, ...]:
        lane = self._index(request_id)
        if self._terminal[lane] or self._pending[lane] is None:
            return ()
        return (self._publish(lane, final=False),)

    def finish(self, request_id: str) -> PublishedAudioChunk:
        lane = self._index(request_id)
        if self._terminal[lane]:
            raise StreamingContractError("request was already published final")
        if self._pending[lane] is None and self.eager and self._published_any[lane]:
            if self.completion_marker:
                marker = PublishedAudioChunk(
                    request_id=self.request_ids[lane],
                    sequence=self._sequence[lane],
                    payload=b"",
                    samples=0,
                    is_final=True,
                )
                self._sequence[lane] += 1
                self._terminal[lane] = True
                return marker
            raise StreamingContractError(
                "eager lane drained before its final chunk: continuation was promised "
                "for a chunk that turned out to be last"
            )
        return self._publish(lane, final=True)


def minimum_streaming_kv_tokens(
    *,
    batch: int,
    prompt_tokens: int,
    max_new_tokens: int = STREAMING_MAX_TOKENS,
) -> int:
    values = (int(batch), int(prompt_tokens), int(max_new_tokens))
    if any(value <= 0 for value in values):
        raise ValueError("batch, prompt_tokens, and max_new_tokens must be positive")
    return values[0] * (values[1] + values[2])


__all__ = [
    "AUDIO_CODEBOOKS",
    "AUDIO_CODEC_TOKEN_MAX",
    "AUDIO_MODEL_TOKEN_MAX",
    "AUDIO_SENTINEL",
    "CHUNK_SCHEDULES",
    "ChunkSchedule",
    "CohortCodeChunk",
    "CumulativeAudioCodeCursor",
    "ExactCohortChunker",
    "MR2_MATCHED_CHUNK_SCHEDULE",
    "MR2_MATCHED_STARTUP_CHUNKS",
    "MR2_MATCHED_STEADY_CHUNK",
    "OPTIMIZED_H100_CHUNK_SCHEDULE",
    "OPTIMIZED_H100_STARTUP_CHUNKS",
    "OPTIMIZED_H100_STEADY_CHUNK",
    "ObservedAudioRow",
    "OneChunkLookaheadPublisher",
    "PublishedAudioChunk",
    "STREAMING_MAX_TOKENS",
    "STREAMING_SPEECH_DELAY",
    "SPEECH_DELAY_ENV",
    "STREAMING_TAP_CAPACITY",
    "SUPPORTED_EXACT_BATCHES",
    "StreamingContractError",
    "ZERO_CODEC_ROW",
    "classify_generated_row",
    "minimum_streaming_kv_tokens",
    "resolved_speech_delay",
    "target_chunk_frames",
]
