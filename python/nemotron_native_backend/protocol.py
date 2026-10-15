from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_COLLECTION_ENTRIES = 16_384
OUTPUT_CONTRACT = "pcm_f32_le_22050_mono_17640"
STREAMING_OUTPUT_CONTRACT = "pcm_f32_le_22050_mono_variable_streaming_v2"
TWO_PHASE_OUTPUT_CONTRACT = "pcm_f32_le_22050_mono_variable_streaming_v3"

CLIENT_KINDS = frozenset({"client_hello", "submit", "abort", "ping"})
WORKER_KINDS = frozenset(
    {"worker_ready", "accepted", "audio", "stream_complete_v1", "failed", "aborted", "pong"}
)


class ProtocolError(ValueError):
    pass


def _pack_length(tag8: int, tag16: int, tag32: int, length: int) -> bytes:
    if length <= 0xFF:
        return bytes((tag8, length))
    if length <= 0xFFFF:
        return bytes((tag16,)) + struct.pack(">H", length)
    if length <= 0xFFFFFFFF:
        return bytes((tag32,)) + struct.pack(">I", length)
    raise ProtocolError("MessagePack value is too large")


def packb(value: Any) -> bytes:

    output = bytearray()

    def emit(item: Any, depth: int = 0) -> None:
        if depth > 64:
            raise ProtocolError("MessagePack nesting exceeds 64")
        if item is None:
            output.append(0xC0)
        elif item is False:
            output.append(0xC2)
        elif item is True:
            output.append(0xC3)
        elif isinstance(item, int):
            if 0 <= item <= 0x7F:
                output.append(item)
            elif -32 <= item < 0:
                output.append(item & 0xFF)
            elif 0 <= item <= 0xFF:
                output.extend((0xCC, item))
            elif 0 <= item <= 0xFFFF:
                output.append(0xCD)
                output.extend(struct.pack(">H", item))
            elif 0 <= item <= 0xFFFFFFFF:
                output.append(0xCE)
                output.extend(struct.pack(">I", item))
            elif 0 <= item <= 0xFFFFFFFFFFFFFFFF:
                output.append(0xCF)
                output.extend(struct.pack(">Q", item))
            elif -(1 << 7) <= item < 0:
                output.append(0xD0)
                output.extend(struct.pack(">b", item))
            elif -(1 << 15) <= item < 0:
                output.append(0xD1)
                output.extend(struct.pack(">h", item))
            elif -(1 << 31) <= item < 0:
                output.append(0xD2)
                output.extend(struct.pack(">i", item))
            elif -(1 << 63) <= item < 0:
                output.append(0xD3)
                output.extend(struct.pack(">q", item))
            else:
                raise ProtocolError("integer is outside MessagePack int64/uint64")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ProtocolError("non-finite floats are forbidden")
            output.append(0xCB)
            output.extend(struct.pack(">d", item))
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            length = len(encoded)
            if length <= 31:
                output.append(0xA0 | length)
            else:
                output.extend(_pack_length(0xD9, 0xDA, 0xDB, length))
            output.extend(encoded)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            encoded = bytes(item)
            output.extend(_pack_length(0xC4, 0xC5, 0xC6, len(encoded)))
            output.extend(encoded)
        elif isinstance(item, Mapping):
            length = len(item)
            if length > MAX_COLLECTION_ENTRIES:
                raise ProtocolError("MessagePack map is unreasonably large")
            if not all(isinstance(key, str) for key in item):
                raise ProtocolError("protocol map keys must be strings")
            if length <= 15:
                output.append(0x80 | length)
            elif length <= 0xFFFF:
                output.append(0xDE)
                output.extend(struct.pack(">H", length))
            else:
                output.append(0xDF)
                output.extend(struct.pack(">I", length))
            for key, child in item.items():
                emit(key, depth + 1)
                emit(child, depth + 1)
        elif isinstance(item, Sequence):
            length = len(item)
            if length > MAX_COLLECTION_ENTRIES:
                raise ProtocolError("MessagePack array is unreasonably large")
            if length <= 15:
                output.append(0x90 | length)
            elif length <= 0xFFFF:
                output.append(0xDC)
                output.extend(struct.pack(">H", length))
            else:
                output.append(0xDD)
                output.extend(struct.pack(">I", length))
            for child in item:
                emit(child, depth + 1)
        else:
            raise ProtocolError(f"unsupported MessagePack type: {type(item).__name__}")

    emit(value)
    return bytes(output)


class _Unpacker:
    def __init__(self, payload: bytes) -> None:
        self.data = memoryview(payload)
        self.offset = 0

    def take(self, size: int) -> memoryview:
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ProtocolError("truncated MessagePack value")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def number(self, fmt: str) -> int | float:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def value(self, depth: int = 0) -> Any:
        if depth > 64:
            raise ProtocolError("MessagePack nesting exceeds 64")
        tag = int(self.take(1)[0])
        if tag <= 0x7F:
            return tag
        if tag >= 0xE0:
            return tag - 256
        if 0xA0 <= tag <= 0xBF:
            return self.string(tag & 0x1F)
        if 0x90 <= tag <= 0x9F:
            return self.array(tag & 0x0F, depth)
        if 0x80 <= tag <= 0x8F:
            return self.mapping(tag & 0x0F, depth)
        if tag == 0xC0:
            return None
        if tag == 0xC2:
            return False
        if tag == 0xC3:
            return True
        if tag == 0xC4:
            return bytes(self.take(int(self.number(">B"))))
        if tag == 0xC5:
            return bytes(self.take(int(self.number(">H"))))
        if tag == 0xC6:
            return bytes(self.take(int(self.number(">I"))))
        if tag == 0xCB:
            value = float(self.number(">d"))
            if not math.isfinite(value):
                raise ProtocolError("non-finite floats are forbidden")
            return value
        numeric = {
            0xCC: ">B",
            0xCD: ">H",
            0xCE: ">I",
            0xCF: ">Q",
            0xD0: ">b",
            0xD1: ">h",
            0xD2: ">i",
            0xD3: ">q",
        }
        if tag in numeric:
            return int(self.number(numeric[tag]))
        if tag == 0xD9:
            return self.string(int(self.number(">B")))
        if tag == 0xDA:
            return self.string(int(self.number(">H")))
        if tag == 0xDB:
            return self.string(int(self.number(">I")))
        if tag == 0xDC:
            return self.array(int(self.number(">H")), depth)
        if tag == 0xDD:
            return self.array(int(self.number(">I")), depth)
        if tag == 0xDE:
            return self.mapping(int(self.number(">H")), depth)
        if tag == 0xDF:
            return self.mapping(int(self.number(">I")), depth)
        raise ProtocolError(f"unsupported MessagePack tag 0x{tag:02x}")

    def array(self, length: int, depth: int) -> list[Any]:
        if length > MAX_COLLECTION_ENTRIES:
            raise ProtocolError("MessagePack array is unreasonably large")
        return [self.value(depth + 1) for _ in range(length)]

    def string(self, length: int) -> str:
        try:
            return bytes(self.take(length)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError("invalid UTF-8 string") from error

    def mapping(self, length: int, depth: int) -> dict[str, Any]:
        if length > MAX_COLLECTION_ENTRIES:
            raise ProtocolError("MessagePack map is unreasonably large")
        result: dict[str, Any] = {}
        for _ in range(length):
            key = self.value(depth + 1)
            if not isinstance(key, str):
                raise ProtocolError("protocol map keys must be strings")
            if key in result:
                raise ProtocolError(f"duplicate protocol map key {key!r}")
            result[key] = self.value(depth + 1)
        return result


def unpackb(payload: bytes) -> Any:
    unpacker = _Unpacker(payload)
    value = unpacker.value()
    if unpacker.offset != len(unpacker.data):
        raise ProtocolError("trailing bytes after MessagePack value")
    return value


def validate_envelope(message: Any, *, from_client: bool) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("protocol frame root must be a map")
    protocol_version = message.get("protocol_version")
    if type(protocol_version) is not int or protocol_version != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol_version")
    kind = message.get("kind")
    allowed = CLIENT_KINDS if from_client else WORKER_KINDS
    if kind not in allowed:
        raise ProtocolError(f"unsupported {'client' if from_client else 'worker'} kind {kind!r}")
    return message


def frame(message: Mapping[str, Any], max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    payload = packb(message)
    if not payload or len(payload) > max_frame_bytes:
        raise ProtocolError(f"frame payload size {len(payload)} is outside allowed range")
    return struct.pack(">I", len(payload)) + payload


async def read_frame(
    reader: asyncio.StreamReader, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> dict[str, Any]:
    header = await reader.readexactly(4)
    length = struct.unpack(">I", header)[0]
    if length == 0 or length > max_frame_bytes:
        raise ProtocolError(f"frame length {length} is outside allowed range")
    value = unpackb(await reader.readexactly(length))
    if not isinstance(value, dict):
        raise ProtocolError("frame root must be a map")
    return value


async def write_frame(
    writer: asyncio.StreamWriter,
    message: Mapping[str, Any],
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    writer.write(frame(message, max_frame_bytes))
    await writer.drain()


def message(kind: str, **fields: Any) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "kind": kind, **fields}
