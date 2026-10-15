import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path

from .. import r31


def install(base, *, incomplete_timeout_seconds=10.0):
    timeout = float(incomplete_timeout_seconds)
    if not math.isfinite(timeout) or not 0.0 < timeout <= 60.0:
        raise ValueError("Invalid full-physical-cohort deadline")
    installed = getattr(base, "_r21_exact_admission_timeout", None)
    if installed is not None:
        if installed != timeout:
            raise ValueError("Exact admission is already installed with another deadline")
        return
    parent = base.TimedBatcher
    error_type = base.BackendError

    class PhysicalWidthBatcher(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self._holding or self.expected_cohort_size != self.backend.batch:
                raise ValueError("Exact admission requires the physical-width holding path")
            if self.expected_cohort_size <= 1 or not isinstance(self.backend.epoch, int):
                raise ValueError("Exact admission requires a wide backend with an epoch")
            self._r21_initial_dispatch = False
            self._r21_incomplete_rejections = 0
            self._r21_incomplete_timeout_seconds = timeout
            self._r31_incremental = r31.enabled("staged_admission")
            self._r31_groups = {}
            self._r31_abort_seen = False
            self._r31_dispatch = {} if r31.enabled("accept_at_dispatch") else None

        def submit(self, request):
            handle = super().submit(request)
            if self._r31_incremental:
                item = self._queue[-1]
                if item.request is not request:
                    raise RuntimeError("submit did not append the admitted request")
                self._r31_groups.setdefault(request.compatibility_key, []).append(item)
            if self._r31_dispatch is not None:
                self._r31_dispatch[request.request_id] = asyncio.Event()
            return handle

        def r31_dispatch_event(self, request_id):
            return None if self._r31_dispatch is None else self._r31_dispatch.get(request_id)

        def abort(self, request_id):
            self._r31_abort_seen = True
            if self._r31_dispatch is not None:
                self._r31_dispatch.pop(request_id, None)
            return super().abort(request_id)

        def _r31_regroup(self):
            groups = {}
            for item in self._queue:
                groups.setdefault(item.request.compatibility_key, []).append(item)
            self._r31_groups = groups

        def _purge_aborted(self):
            if not self._r31_incremental:
                return super()._purge_aborted()
            if self._r31_abort_seen:
                self._r31_abort_seen = False
                super()._purge_aborted()
                self._r31_regroup()

        def _grouped_queue(self):
            if not self._r31_incremental:
                return super()._grouped_queue()
            return {key: items for (key, items) in self._r31_groups.items() if items}

        def _reject_unrepresentable(self, items):
            super()._reject_unrepresentable(items)
            if self._r31_incremental:
                self._r31_regroup()

        def _finish_queued_error(self, item, error):
            self.backend.arrivals.pop(item.request.request_id, None)
            if self._r31_dispatch is not None:
                self._r31_dispatch.pop(item.request.request_id, None)
            return super()._finish_queued_error(item, error)

        def _reject_incomplete_physical_cohort(self, items):
            items = list(items)
            error = error_type(
                "incomplete_physical_cohort",
                f"{len(items)} compatible request(s) did not form physical "
                f"B{self.expected_cohort_size} within {timeout:g} seconds; "
                "partial neural execution is disabled",
                retryable=True,
            )
            chosen = {id(item) for item in items}
            self._queue = deque(item for item in self._queue if id(item) not in chosen)
            for item in items:
                self._finish_queued_error(item, error)
            self._r21_incomplete_rejections += len(items)
            if self._r31_incremental:
                self._r31_regroup()
            receipt = dict(
                code="incomplete_physical_cohort",
                request_ids=[item.request.request_id for item in items],
                epoch=self.backend.epoch,
                pending_after=len(self._pending),
                arrivals_after=len(self.backend.arrivals),
                monotonic_ns=time.monotonic_ns(),
            )
            with (Path(self.backend.receipt_dir) / "r21-incomplete-cohorts.jsonl").open(
                "a"
            ) as stream:
                stream.write(json.dumps(receipt, separators=(",", ":")) + "\n")

        def _plan_cohorts(self, groups, now):
            groups = list(groups.values())
            if (
                self.backend.epoch == 0
                and not self._r21_initial_dispatch
                and self._committed_batches == 0
                and len(groups) == 1
                and len(groups[0]) == 1
            ):
                return super()._plan_cohorts(
                    {groups[0][0].request.compatibility_key: groups[0]}, now
                )
            plan = []
            waits = []
            committed = self._committed_batches
            expected = self.expected_cohort_size
            for items in groups:
                if not items:
                    continue
                if len(items) >= expected:
                    if committed < self.max_inflight_batches:
                        plan.append(items[:expected])
                        committed += 1
                    continue
                remaining = timeout - (now - items[0].enqueued_at)
                if remaining <= 0.0:
                    self._reject_incomplete_physical_cohort(items)
                else:
                    waits.append(remaining)
            return plan, None if plan else min(waits) if waits else None

        def _launch_cohort(self, cohort):
            readiness = (
                len(cohort) == 1
                and self.backend.epoch == 0
                and not self._r21_initial_dispatch
                and self._committed_batches == 0
            )
            if len(cohort) != self.expected_cohort_size and not readiness:
                raise RuntimeError("Exact admission refused a partial neural cohort")
            self._r21_initial_dispatch = True
            result = super()._launch_cohort(cohort)
            if self._r31_incremental:
                self._r31_regroup()
            if self._r31_dispatch is not None:
                for item in cohort:
                    event = self._r31_dispatch.pop(item.request.request_id, None)
                    if event is not None:
                        event.set()
            return result

    base.TimedBatcher = PhysicalWidthBatcher
    base._r21_exact_admission_timeout = timeout
