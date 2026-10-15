import asyncio
import hashlib
import json
import threading
import time

REGISTRY = {}


class FirstFence:
    def __init__(self, request_ids, timeout=0.050):
        if len(set(request_ids)) != len(request_ids) or not 0 < timeout <= 0.050:
            raise ValueError("Invalid first-audio fence contract")
        self.ids = tuple(request_ids)
        self.timeout = timeout
        self.condition = threading.Condition()
        self.owned = set()
        self.retired = set()
        self.used = False
        self.closed = False
        self.waits = []

    def acknowledge(self, request_id):
        with self.condition:
            if request_id not in self.ids:
                raise ValueError("ACK belongs to another cohort")
            self.owned.add(request_id)
            self.condition.notify_all()

    def retire(self, request_id):
        with self.condition:
            if request_id in self.ids:
                self.retired.add(request_id)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def wait(self, lanes):
        if self.used:
            return
        required = {self.ids[lane] for lane in lanes}
        if not required:
            return
        self.used = True
        begin = time.monotonic_ns()
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while not self.closed and not required <= self.owned | self.retired:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            self.waits.append(
                dict(
                    requests=len(required),
                    start_ns=begin,
                    end_ns=time.monotonic_ns(),
                    missing=len(required - self.owned - self.retired),
                    closed=self.closed,
                )
            )

    def receipt(self):
        return dict(
            policy="once-after-original-continuation-before-G9-frontend-owned-ACK",
            maximum_wait_ms=self.timeout * 1000,
            waits=list(self.waits),
            acknowledged=len(self.owned),
            retired=len(self.retired),
            public_http_ack=False,
            requests_sha256=hashlib.sha256(json.dumps(self.ids).encode()).hexdigest(),
        )


def acknowledge(request_id):
    gate = REGISTRY.get(request_id)
    if gate is not None:
        gate.acknowledge(request_id)


def retire(request_id):
    gate = REGISTRY.get(request_id)
    if gate is not None:
        gate.retire(request_id)


def install_backend(base):
    previous = base.NativeBackend.stream_batch

    async def stream_batch(self, requests, emit):
        if not hasattr(self, "_r21_fence_owner"):
            self._r21_fence_owner = asyncio.Lock()
        async with self._r21_fence_owner:
            target = self.engines[len(requests)]
            gate = FirstFence([request.request_id for request in requests])
            if any(request_id in REGISTRY for request_id in gate.ids):
                raise RuntimeError("Duplicate active fence request")
            target._r21_first_fence = gate
            REGISTRY.update({request_id: gate for request_id in gate.ids})
            try:
                return await previous(self, requests, emit)
            finally:
                gate.close()
                target._r21_first_fence = None
                for request_id in gate.ids:
                    REGISTRY.pop(request_id, None)

    base.NativeBackend.stream_batch = stream_batch


def install_engine(engine):
    previous = engine.stream

    def stream(*args, **kwargs):
        iterator = previous(*args, **kwargs)
        try:
            for event in iterator:
                gate = getattr(engine, "_r21_first_fence", None)
                if gate is not None and event["type"] == "receipt":
                    event["receipt"]["r21_first_fence"] = gate.receipt()
                yield event
                if gate is not None and event["type"] == "continuation":
                    gate.wait(event["lanes"])
        finally:
            iterator.close()

    engine.stream = stream
