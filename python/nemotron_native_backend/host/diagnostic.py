import json
import threading
import time
from pathlib import Path


class Trace:
    def __init__(self, path=None):
        self.stream = None
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.stream = target.open("a", buffering=1)
        self.lock = threading.Lock()

    def event(self, name, **fields):
        if self.stream is not None:
            with self.lock:
                self.stream.write(
                    json.dumps(
                        dict(
                            event=name,
                            monotonic_ns=time.monotonic_ns(),
                            thread_cpu_ns=time.thread_time_ns(),
                            thread=threading.get_native_id(),
                            **fields,
                        )
                    )
                    + "\n"
                )


TRACE = Trace()
