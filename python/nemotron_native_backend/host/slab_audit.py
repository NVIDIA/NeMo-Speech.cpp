import hashlib
import json


def pcm16(payload):
    import numpy as np

    x = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite PCM")

    scaled = np.where(x < 0, x * np.float32(32768), x * np.float32(32767))
    y = np.copysign(np.floor(np.abs(scaled.astype(np.float64)) + 0.5), scaled)
    y = np.where(x <= -1, -32768, np.where(x >= 1, 32767, y))
    return y.astype("<i2").tobytes()


class SlabAudit:
    def __init__(self, path):
        self.stream = path.open("x")
        self.requests = {}

    def audio(self, message):
        ident = message["request_id"]
        row = self.requests.setdefault(ident, dict(sequence=0, samples=0, sha=hashlib.sha256()))
        if message["sequence"] != row["sequence"]:
            raise ValueError("audit sequence mismatch")
        row["sha"].update(pcm16(message["data"]))
        row["sequence"] += 1
        row["samples"] += message["samples"]
        if message["is_final"]:
            self.stream.write(
                json.dumps(
                    dict(
                        request_id=ident,
                        samples=row["samples"],
                        chunks=row["sequence"],
                        pcm16_sha256=row["sha"].hexdigest(),
                        status="FINAL",
                    )
                )
                + "\n"
            )
            self.stream.flush()
            del self.requests[ident]

    def close(self):
        for ident, row in self.requests.items():
            self.stream.write(
                json.dumps(
                    dict(
                        request_id=ident,
                        samples=row["samples"],
                        chunks=row["sequence"],
                        pcm16_sha256=row["sha"].hexdigest(),
                        status="ABANDONED",
                    )
                )
                + "\n"
            )
        self.stream.close()
