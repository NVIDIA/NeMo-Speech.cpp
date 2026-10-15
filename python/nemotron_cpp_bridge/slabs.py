# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import mmap
import os
import re
import stat


class SlabReader:
    SLOT_BYTES = 8 * 1764 * 4

    def __init__(self, directory="/dev/shm"):
        self.directory = directory
        self.arena = None
        self.mapping = None
        self.generations = {}

    def copy(self, value):
        integer_fields = (
            "slab_version",
            "slot",
            "generation",
            "arena_bytes",
            "slot_bytes",
            "offset",
            "data_bytes",
            "samples",
            "sequence",
            "sample_rate_hz",
            "channels",
        )
        if any(type(value.get(key)) is not int for key in integer_fields):
            raise ValueError("slab metadata requires integer fields")
        arena = value.get("arena")
        if not isinstance(arena, str) or re.fullmatch(r"r20-pcm-[0-9a-f]{32}", arena) is None:
            raise ValueError("invalid PCM arena name")
        extent = value["arena_bytes"]
        slot = value["slot"]
        size = value["data_bytes"]
        if (
            value["slab_version"] != 1
            or value["slot_bytes"] != self.SLOT_BYTES
            or not 0 < extent <= 4096 * self.SLOT_BYTES
            or extent % self.SLOT_BYTES
            or not 0 <= slot < extent // self.SLOT_BYTES
            or value["generation"] <= self.generations.get(slot, 0)
            or value["offset"] != slot * self.SLOT_BYTES
            or not 0 < size <= self.SLOT_BYTES
            or size != value["samples"] * 4
            or value["offset"] + size > extent
            or value["sample_rate_hz"] != 22050
            or value["channels"] != 1
            or value.get("encoding") != "pcm_f32_le"
            or value.get("is_final") is not False
            or not 0 <= value["sequence"] <= 0xFFFFFFFF
        ):
            raise ValueError("invalid PCM slab bounds or audio contract")
        if self.mapping is None:
            fd = os.open(
                os.path.join(self.directory, arena), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size != extent:
                    raise ValueError("PCM arena file does not match its descriptor")
                self.mapping = mmap.mmap(fd, extent, access=mmap.ACCESS_READ)
                self.arena = arena
            finally:
                os.close(fd)
        if arena != self.arena or len(self.mapping) != extent:
            raise ValueError("PCM arena identity changed on an active connection")
        self.generations[slot] = value["generation"]
        return self.mapping[value["offset"] : value["offset"] + size]

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
