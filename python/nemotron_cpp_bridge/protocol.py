# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import math
import struct

try:
    import numpy as np
except ImportError:
    np = None

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_CONTROL_FRAME_BYTES = 8 * 1024 * 1024


def control_frame(kind, **fields):
    payload = b"J" + json.dumps(
        {"kind": kind, "protocol_version": 1, **fields},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return frame(payload)


def frame(payload):
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("bridge frame is outside the allowed size")
    return struct.pack(">I", len(payload)) + payload


def audio_frame(request_id, sequence, pcm):
    if not pcm or len(pcm) % 2:
        raise ValueError("invalid PCM16 extent")
    return frame(b"A" + struct.pack(">QI", request_id, sequence) + pcm)


def float_audio_frame(request_id, sequence, pcm):
    if not pcm or len(pcm) % 4:
        raise ValueError("invalid PCM-F32LE extent")
    return frame(b"F" + struct.pack(">QI", request_id, sequence) + pcm)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


async def read_control(reader):
    length = struct.unpack(">I", await reader.readexactly(4))[0]
    if not 1 < length <= min(MAX_FRAME_BYTES, MAX_CONTROL_FRAME_BYTES):
        raise ValueError("invalid control frame length")
    payload = await reader.readexactly(length)
    if payload[:1] != b"J":
        raise ValueError("client must send JSON control frames")
    value = json.loads(
        payload[1:],
        object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON value")),
    )
    if not isinstance(value, dict) or value.get("protocol_version", 1) != 1:
        raise ValueError("unsupported bridge protocol")
    return value


def pcm16_scalar(payload):
    if len(payload) % 4:
        raise ValueError("invalid PCM-F32LE extent")
    output = bytearray(len(payload) // 2)
    for index, (sample,) in enumerate(struct.iter_unpack("<f", payload)):
        if not math.isfinite(sample):
            raise ValueError("nonfinite PCM sample")
        if sample <= -1:
            value = -32768
        elif sample >= 1:
            value = 32767
        else:
            scale = 32768 if sample < 0 else 32767
            scaled = struct.unpack("<f", struct.pack("<f", sample * scale))[0]
            magnitude = math.floor(abs(scaled) + 0.5)
            value = -magnitude if scaled < 0 else magnitude
        struct.pack_into("<h", output, index * 2, value)
    return bytes(output)


def pcm16(payload):
    if np is None:
        return pcm16_scalar(payload)
    if len(payload) % 4:
        raise ValueError("invalid PCM-F32LE extent")
    samples = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite PCM sample")
    clipped = np.clip(samples, -1, 1)
    scaled = np.where(clipped < 0, clipped * np.float32(32768), clipped * np.float32(32767))
    rounded = np.copysign(np.floor(np.abs(scaled.astype(np.float64)) + 0.5), scaled)
    return rounded.astype("<i2").tobytes()
