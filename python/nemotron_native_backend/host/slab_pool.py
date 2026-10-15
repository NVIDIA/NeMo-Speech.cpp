import asyncio
import collections
import mmap
import os
import secrets


class SlabPool:
    SCHEMA = 1
    SLOT_BYTES = 8 * 1764 * 4

    def __init__(self, slots=128, directory="/dev/shm"):
        if not 1 <= slots <= 4096:
            raise ValueError("slab slot count must be bounded")
        self.arena = "r20-pcm-" + secrets.token_hex(16)
        self.path = os.path.join(directory, self.arena)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        self.bytes = slots * self.SLOT_BYTES
        try:
            os.posix_fallocate(self.fd, 0, self.bytes)
            self.mapping = mmap.mmap(self.fd, self.bytes)
        except BaseException:
            os.close(self.fd)
            os.unlink(self.path)
            raise
        self.slots = slots
        self.available = asyncio.Queue(maxsize=slots)
        for slot in range(slots):
            self.available.put_nowait(slot)
        self.generations = [0] * slots
        self.outstanding = {}
        self.closed = False
        self.waiters = set()
        self.counts = collections.Counter()

    async def publish(self, payload):
        if self.closed:
            raise ConnectionError("PCM arena retired")
        if (
            type(payload) is not bytes
            or not 0 < len(payload) <= self.SLOT_BYTES
            or len(payload) % 4
        ):
            raise ValueError("invalid PCM float payload size/type")
        try:
            slot = self.available.get_nowait()
        except asyncio.QueueEmpty:
            waiter = asyncio.create_task(self.available.get())
            self.waiters.add(waiter)
            try:
                slot = await waiter
            except asyncio.CancelledError:
                if waiter.done() and not waiter.cancelled() and not self.closed:
                    self.available.put_nowait(waiter.result())
                raise
            finally:
                self.waiters.discard(waiter)
        if self.closed:
            raise ConnectionError("PCM arena retired while waiting")
        self.generations[slot] += 1
        generation = self.generations[slot]
        offset = slot * self.SLOT_BYTES
        self.mapping[offset : offset + len(payload)] = payload
        self.outstanding[slot] = generation
        self.counts["published"] += 1
        self.counts["float_bytes_via_slab"] += len(payload)
        self.counts["max_outstanding"] = max(self.counts["max_outstanding"], len(self.outstanding))
        return dict(
            slab_version=1,
            arena=self.arena,
            slot=slot,
            generation=generation,
            arena_bytes=self.bytes,
            slot_bytes=self.SLOT_BYTES,
            offset=offset,
            data_bytes=len(payload),
        )

    def acknowledge(self, arena, slot, generation):
        if (
            self.closed
            or arena != self.arena
            or type(slot) is not int
            or type(generation) is not int
        ):
            raise ValueError("invalid PCM lease acknowledgement")
        if self.outstanding.get(slot) != generation:
            raise ValueError("stale, duplicate or unknown PCM lease acknowledgement")
        del self.outstanding[slot]
        self.available.put_nowait(slot)
        self.counts["acknowledged"] += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        for waiter in tuple(self.waiters):
            waiter.cancel()
        self.counts["retired_unacknowledged"] = len(self.outstanding)
        self.mapping.close()
        os.close(self.fd)

        os.unlink(self.path)

    def receipt(self):
        return dict(
            schema="r20.pcm-slab.v1",
            arena=self.arena,
            slots=self.slots,
            slot_bytes=self.SLOT_BYTES,
            bytes=self.bytes,
            closed=self.closed,
            outstanding=len(self.outstanding),
            **self.counts,
        )
