from __future__ import annotations

import gc
import json
import os
import time

KNOWN = (
    "gc_trace",
    "gc_quiet",
    "native_unpack",
    "staged_admission",
    "eager_publication",
    "fast_first_chunk",
    "rng_predraw",
    "accept_at_dispatch",
)


def _parse():
    raw = os.environ.get("NEMOTRON_R31", "{}")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) - set(KNOWN)
        or not all(type(v) is bool for v in value.values())
    ):
        raise RuntimeError("NEMOTRON_R31 must be a JSON object of known boolean options: " + raw)
    return {key: bool(value.get(key, key != "gc_trace")) for key in KNOWN}


OPTIONS = _parse()


def enabled(name):
    return OPTIONS[name]


class GcTrace:
    def __init__(self):
        self.events = []
        self._open = {}

    def __call__(self, phase, info):
        now = time.monotonic_ns()
        generation = info.get("generation")
        if phase == "start":
            self._open[generation] = now
        else:
            self.events.append(
                (
                    self._open.pop(generation, now),
                    now,
                    generation,
                    info.get("collected", 0),
                    info.get("uncollectable", 0),
                )
            )

    def drain(self):
        events, self.events = self.events, []
        return [
            dict(start_ns=a, end_ns=b, generation=g, collected=c, uncollectable=u, ms=(b - a) / 1e6)
            for (a, b, g, c, u) in events
        ]


TRACE = GcTrace()
_installed = False


def after_prepare():
    global _installed
    if _installed:
        return
    _installed = True
    if OPTIONS["gc_trace"] or OPTIONS["gc_quiet"]:
        gc.callbacks.append(TRACE)
    if OPTIONS["gc_quiet"]:
        gc.collect()
        gc.freeze()
        gc.disable()


def cohort_boundary():
    if OPTIONS["gc_quiet"]:
        gc.collect()


def receipt():
    return dict(
        options=dict(OPTIONS),
        gc_enabled=gc.isenabled(),
        gc_frozen=gc.get_freeze_count(),
        gc_thresholds=list(gc.get_threshold()),
        gc_events=TRACE.drain(),
    )
