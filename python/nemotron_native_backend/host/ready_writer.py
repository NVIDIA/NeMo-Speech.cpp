import asyncio


class ReadyWriter:
    def __init__(self, write, capacity=4096, group=64):
        if not 1 <= group <= 64 or capacity < group:
            raise ValueError("Invalid ready-writer bounds")
        self.write = write
        self.queue = asyncio.Queue(maxsize=capacity)
        self.group = group
        self.closed = False
        self.failure = None
        self.peak = 0
        self.frames = 0
        self.records = 0
        self.maximum_group = 0
        self.task = asyncio.create_task(self.run())

    async def send(self, value):
        if self.closed:
            raise ConnectionError("Ready writer closed") from self.failure
        future = asyncio.get_running_loop().create_future()
        put = None
        try:
            try:
                self.queue.put_nowait((value, future))
            except asyncio.QueueFull:
                put = asyncio.create_task(self.queue.put((value, future)))
                done, _ = await asyncio.wait((put, self.task), return_when=asyncio.FIRST_COMPLETED)
                if self.task in done:
                    raise ConnectionError("Ready writer stopped") from self.failure
                await put
            self.peak = max(self.peak, self.queue.qsize())
            await future
        finally:
            if put is not None and not put.done():
                put.cancel()
                await asyncio.gather(put, return_exceptions=True)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def run(self):
        pending = None
        active = []
        try:
            while True:
                first = pending if pending is not None else await self.queue.get()
                pending = None
                active = [first]
                if first[0].get("kind") == "audio_slab":
                    while len(active) < self.group:
                        try:
                            following = self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if following[0].get("kind") != "audio_slab":
                            pending = following
                            break
                        active.append(following)
                value = (
                    active[0][0]
                    if len(active) == 1
                    else dict(
                        kind="audio_slab_batch",
                        protocol_version=1,
                        records=[item[0] for item in active],
                    )
                )
                await self.write(value)
                self.frames += 1
                self.records += len(active)
                self.maximum_group = max(self.maximum_group, len(active))
                for _, future in active:
                    if not future.done():
                        future.set_result(None)
                active = []
        except BaseException as error:
            self.failure = error
        finally:
            self.closed = True
            if pending is not None:
                active.append(pending)
            while not self.queue.empty():
                active.append(self.queue.get_nowait())
            for _, future in active:
                if not future.done():
                    future.set_exception(ConnectionError("Ready writer stopped"))

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def ack_records(incoming):
    if incoming.get("kind") == "audio_ack":
        records = [incoming]
        if set(incoming) != {"kind", "protocol_version", "arena", "slot", "generation"}:
            raise ValueError("Invalid single ACK")
    elif incoming.get("kind") == "audio_ack_batch":
        if set(incoming) != {"kind", "protocol_version", "records"}:
            raise ValueError("Invalid ACK batch")
        records = incoming["records"]
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise ValueError("Invalid ACK batch extent")
        if any(
            not isinstance(item, dict) or set(item) != {"arena", "slot", "generation"}
            for item in records
        ):
            raise ValueError("Invalid ACK record")
    else:
        raise ValueError("Not an ACK frame")
    if incoming["protocol_version"] != 1:
        raise ValueError("Invalid ACK version")
    if len({(item["arena"], item["slot"], item["generation"]) for item in records}) != len(records):
        raise ValueError("Duplicate ACK in one frame")
    return records
