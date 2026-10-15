from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import protocol, r31
from .backend import BackendError, BackendIdentity
from .bundle import BundleError
from .execution.composition import create_for_qualification
from .execution.engine import _CONSTRUCTION_LOCK
from .hardware import validate_native_device
from .host.cpu_pools import configure, snapshot
from .host.diagnostic import Trace
from .host.exact_admission import install as install_exact_admission
from .host.serialization import install as install_serialization
from .host.transport import SlabServerMixin
from .server import WorkerServer
from .streaming_batcher import StreamingExactBatcher
from .streaming_contract import OneChunkLookaheadPublisher
from .streaming_models import StreamingAudioChunk, StreamingResult
from .streaming_server import StreamingWorkerServer

CONTRACT = "native_full_prefill_full_stage0_full_stage1_streaming_v1"


class NativeManifest:
    def __init__(self, plan):
        self.plan = plan
        self.sha256 = plan.manifest_sha256
        self.enabled_buckets = tuple(sorted({1, plan.physical_width}))
        compatibility = plan.contract["transport_compatibility"]
        self.raw = dict(
            execution_contract=compatibility["execution_contract"],
            model_sha256=plan.contract["model"]["checkpoint_sha256"],
            output_contract=compatibility["output_contract"],
            installed_transport_versions=dict(compatibility["installed_transport_versions"]),
        )
        if self.raw["execution_contract"] != CONTRACT:
            raise BundleError("Native backend contract mismatch")

    def worker_ready_fields(self):
        return dict(
            manifest_sha256=self.sha256,
            model_sha256=self.raw["model_sha256"],
            enabled_buckets=list(self.enabled_buckets),
            output_contract=self.raw["output_contract"],
            **self.raw["installed_transport_versions"],
        )


class NativeServer(StreamingWorkerServer):
    def __init__(self, **kwargs):

        if kwargs["backend"].identity.execution_contract != CONTRACT:
            raise RuntimeError("native backend contract mismatch")
        WorkerServer.__init__(self, **kwargs)


class TimedBatcher(StreamingExactBatcher):
    def submit(self, request):
        observed = time.monotonic_ns()
        handle = super().submit(request)
        self.backend.arrivals[request.request_id] = observed
        return handle


class BaseNativeBackend:
    def __init__(self, manifest, batch, receipt_dir, *, engine_factory, delivery_group):
        import torch

        hardware = validate_native_device(torch.cuda, receipt_dir)
        self.engine_factory = engine_factory
        self.delivery_group = delivery_group
        self.manifest = manifest
        self.batch = batch
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-cohort")
        self._shutdown_task = None
        self.engines = {}
        self.epoch = 0
        self.arrivals = {}
        self.owner = asyncio.Lock()
        self._identity = BackendIdentity(
            hardware["detected"]["name"],
            tuple(hardware["detected"]["compute_capability"]),
            manifest.enabled_buckets,
            True,
            CONTRACT,
        )

    @property
    def identity(self):
        return self._identity

    async def prepare(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self._prepare)

    async def shutdown(self):
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self):
        def close():
            error = None
            inflight = getattr(self, "inflight_engine", None)
            if inflight is not None:
                try:
                    inflight.close()
                except BaseException as exc:
                    error = exc
                self.engines.pop(self.batch, None)
            for engine in self.engines.values():
                try:
                    engine.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error

        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, close)
        finally:
            self.executor.shutdown(wait=True)

    async def _stream_batch_impl(self, requests, emit_chunk):
        group_size = self.delivery_group
        deferred_audit = False
        async with self.owner:
            self.epoch += 1
            epoch = self.epoch
            batch = len(requests)
            audit_enabled = deferred_audit and batch == 512
            if any(r.seed != requests[0].seed for r in requests):
                raise BackendError("seed_contract", "native cohort requires a shared base seed")
            loop = asyncio.get_running_loop()
            request_ids = [r.request_id for r in requests]
            lanes = {request_id: lane for (lane, request_id) in enumerate(request_ids)}
            if len(lanes) != batch:
                raise RuntimeError("request identities must be unique")
            eager = r31.enabled("eager_publication")
            if (
                eager
                and self.manifest.raw["output_contract"]
                != "pcm_f32_le_22050_mono_variable_streaming_v3"
            ):
                raise BundleError("Eager publication requires the streaming_v3 completion message")
            publisher = OneChunkLookaheadPublisher(
                request_ids, eager=eager, completion_marker=eager
            )
            counts = [0] * batch
            samples = [0] * batch
            first_public = [None] * batch
            start = time.perf_counter()
            start_ns = time.monotonic_ns()
            captured = {}
            handoffs = []
            abandoned = threading.Event()
            pending_future = []
            published_since_yield = 0
            fairness_yields = 0

            async def publish(chunk):
                nonlocal published_since_yield, fairness_yields
                lane = lanes[chunk.request_id]
                audio = StreamingAudioChunk(
                    chunk.request_id, chunk.sequence, chunk.payload, chunk.samples, chunk.is_final
                )
                await emit_chunk(audio)
                counts[lane] += 1
                samples[lane] += audio.samples
                if first_public[lane] is None:
                    first_public[lane] = (time.perf_counter() - start) * 1000
                if audit_enabled:
                    audit = getattr(self.engines[batch], "_r11_audit", None)
                    if audit is None:
                        raise RuntimeError("deferred audit requires installed engine")
                    await audit.published(lane, chunk.payload)
                published_since_yield += 1
                if group_size and published_since_yield == group_size:
                    published_since_yield = 0
                    fairness_yields += 1
                    await asyncio.sleep(0)

            async def consume(events):
                for event in events:
                    kind = event["type"]
                    if kind == "chunk":
                        chunks = publisher.decoded(
                            request_ids[event["lane"]], event["pcm_f32_le"], event["samples"]
                        )
                        for chunk in chunks:
                            await publish(chunk)
                    elif kind == "continuation":
                        for lane in event["lanes"]:
                            for chunk in publisher.observe_codec_continuation(request_ids[lane]):
                                await publish(chunk)
                    elif kind == "finish":
                        await publish(publisher.finish(request_ids[event["lane"]]))
                    elif kind == "receipt":
                        captured.update(event["receipt"])
                    else:
                        raise RuntimeError("unexpected native event: " + kind)

            def handoff(events):
                if abandoned.is_set():
                    raise RuntimeError("cohort delivery abandoned")
                begin = time.monotonic_ns()
                future = asyncio.run_coroutine_threadsafe(consume(events), loop)
                pending_future.append(future)
                try:
                    if abandoned.is_set():
                        future.cancel()
                    future.result()
                finally:
                    pending_future.remove(future)
                handoffs.append(
                    {
                        "kind": events[0]["type"],
                        "events": len(events),
                        "start_ns": begin,
                        "end_ns": time.monotonic_ns(),
                    }
                )

            def produce():
                if batch not in self.engines:
                    self.engines[batch] = self.engine_factory(batch)
                if audit_enabled:
                    raise BundleError("Deferred audit is not part of the imported host contract")
                events = self.engines[batch].stream(
                    [r.normalized_text for r in requests],
                    token_ids=[r.token_ids for r in requests],
                    speaker=requests[0].speaker_id,
                    seed=requests[0].seed,
                    epoch=epoch,
                )
                buffered = []
                try:
                    for event in events:
                        if event["type"] == "chunk":
                            buffered.append(event)
                            if len(buffered) > batch:
                                raise RuntimeError("bounded callback batch exceeded")
                            if event.get("batch_end", True):
                                handoff(tuple(buffered))
                                buffered.clear()
                        else:
                            if buffered:
                                raise RuntimeError("chunk group missing terminal marker")
                            handoff((event,))
                finally:
                    events.close()

            producer = loop.run_in_executor(self.executor, produce)
            try:
                await asyncio.shield(producer)
            except BaseException as error:
                abandoned.set()
                for future in tuple(pending_future):
                    future.cancel()
                while not producer.done():
                    try:
                        await asyncio.shield(producer)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                try:
                    producer.result()
                except BaseException:
                    pass
                (self.receipt_dir / f"cohort-{epoch:05d}-FAILED.json").write_text(
                    json.dumps(dict(epoch=epoch, batch=batch, error=repr(error)), indent=2) + "\n"
                )
                raise
            captured.update(
                execution_contract=CONTRACT,
                epoch=epoch,
                batch=batch,
                backend_started_monotonic_ns=start_ns,
                backend_wall_ms=(time.perf_counter() - start) * 1000,
                first_public_enqueue_ms=first_public,
                r8_handoffs=handoffs,
                r11_delivery=dict(
                    group_size=group_size,
                    fairness_yields=fairness_yields,
                    deferred_audit=audit_enabled,
                ),
                requests=[
                    dict(
                        request_id=r.request_id,
                        normalized_text=r.normalized_text,
                        normalized_text_sha256=hashlib.sha256(
                            r.normalized_text.encode()
                        ).hexdigest(),
                        token_ids=list(r.token_ids),
                        seed=r.seed,
                        emitted_samples=samples[i],
                        submit_monotonic_ns=self.arrivals.pop(r.request_id, None),
                        published_chunks=counts[i],
                    )
                    for (i, r) in enumerate(requests)
                ],
            )
            captured["r31"] = r31.receipt()
            (self.receipt_dir / f"cohort-{epoch:05d}.json").write_text(
                json.dumps(captured, indent=2) + "\n"
            )
            r31.cohort_boundary()
            return [
                StreamingResult(r.request_id, counts[i], samples[i])
                for (i, r) in enumerate(requests)
            ]


class NativeBackend(BaseNativeBackend):
    async def stream_batch(self, requests, emit_chunk):
        if self.epoch >= 1 and len(requests) != self.batch:
            arrival_values = list(self.arrivals.values())
            row = dict(
                status="REJECTED_NONWIDE_DISPATCH_BEFORE_NEURAL_EXECUTION",
                expected=self.batch,
                selected=len(requests),
                epoch=self.epoch,
                outstanding_arrivals=len(arrival_values),
                observed_monotonic_ns=time.monotonic_ns(),
                oldest_submit_ns=min(arrival_values) if arrival_values else None,
                latest_submit_ns=max(arrival_values) if arrival_values else None,
                startup_request_only_allows_b1=True,
            )
            path = self.receipt_dir / "route-rejection.json"
            if not path.exists():
                path.write_text(json.dumps(row, indent=2) + "\n")
            raise RuntimeError(
                "Native wide HTTP refuses post-readiness width " + str(len(requests))
            )
        if self.epoch == 1 and len(requests) == self.batch:
            values = [self.arrivals[r.request_id] for r in requests]
            row = dict(
                status="FIRST_EXCLUDED_WIDE_WAVE_ADMITTED",
                physical_batch=self.batch,
                request_count=len(requests),
                arrival_span_ms=(max(values) - min(values)) / 1000000.0,
                latest_to_dispatch_ms=(time.monotonic_ns() - max(values)) / 1000000.0,
                cohort_timeout_ms=10000,
                readiness_timeout_ms=1000,
                epoch_before_dispatch=self.epoch,
            )
            (self.receipt_dir / "first-wide-dispatch.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
        return await self._stream_batch_impl(requests, emit_chunk)

    def _prepare(self):
        mode = self.manifest.plan.contract["host"].get("batching_mode", "fixed_cohort")
        widths = {1, self.batch} if mode == "fixed_cohort" else {self.batch}
        for width in sorted(widths):
            engine = self.engine_factory(width)
            engine.retain_codes = False
            self.engines[width] = engine
            events = engine.stream(
                ["Hello from the native three megakernel runner."] * width, seed=0, epoch=1
            )
            counts = {"chunk": 0, "continuation": 0, "finish": 0, "receipt": 0}
            try:
                for event in events:
                    counts[event["type"]] += 1
            finally:
                events.close()
            if counts["finish"] != width or counts["receipt"] != 1:
                raise RuntimeError("Wide HTTP startup failed publication lifecycle")
            receipt = dict(
                status="PASS_EXECUTION_INTEGRITY",
                batch=width,
                events=counts,
                native=engine.last_receipt,
                source_contract=self.manifest.sha256,
                scope="Prepared before HTTP; fresh width-specific execution qualification "
                "required by launch seal",
                cpu_pools=snapshot(self),
            )
            (self.receipt_dir / f"http-preparation-b{width}.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        if mode != "fixed_cohort":
            from .execution.inflight_engine import InflightEngine

            if mode not in ("fixed_request", "inflight"):
                raise BundleError("Unsupported request batching mode")
            if not r31.enabled("eager_publication"):
                raise BundleError(
                    "In-flight execution requires the retained eager publication path"
                )
            engine = self.engines[self.batch]
            library = engine.native_artifacts.get("stage0_inflight")
            if library is None:
                raise BundleError("Missing qualified source-bound in-flight producer")
            self.inflight_engine = InflightEngine.from_engine(
                engine,
                stage0_library=library,
                policy=self.manifest.plan.contract["host"].get("inflight_policy"),
            )
            (self.receipt_dir / "inflight-binding.json").write_text(
                json.dumps(
                    dict(
                        mode=mode,
                        physical_batch=self.batch,
                        request_stable_rng=True,
                        contraction=False,
                        stage0_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
                + "\n"
            )


class NativeSlabServer(SlabServerMixin, NativeServer):
    def __init__(self, *, host_options, slab_audit_enabled=False, trace=None, **kwargs):
        if dict(host_options) != {"f5": True}:
            raise BundleError("Only the selected f5 host transport is ported")
        self.native_host_options = dict(host_options)
        self.native_dma = None
        self.slab_audit_enabled = bool(slab_audit_enabled)
        self.native_trace = trace if trace is not None else Trace()
        super().__init__(**kwargs)


@dataclass(frozen=True)
class ReadinessBinding:
    factory: Callable[[], object]
    source_sha256: str
    physical_width: int = 1


@dataclass(frozen=True)
class QualificationWorker:
    backend: NativeBackend
    batcher: object
    server: NativeSlabServer
    readiness_source_sha256: str


async def prepare_for_qualification(
    plan,
    artifact_root,
    model_root,
    *,
    source_pins,
    model_identity,
    readiness: ReadinessBinding,
    socket_path,
    receipts,
    qualification_only,
    slab_audit_enabled=False,
    trace=None,
    engine_factory_override=None,
):
    if qualification_only is not True or plan.contract["enabled"] is not False:
        raise BundleError("Production native package serving is not enabled")
    if (
        readiness.physical_width != 1
        or len(readiness.source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in readiness.source_sha256)
    ):
        raise BundleError("A separately pinned physical-B1 readiness factory is required")
    from .execution.readiness import ReadinessFactory

    if type(readiness.factory) is not ReadinessFactory:
        raise BundleError("Readiness requires the package source-verified factory")
    readiness.factory.verify_identity(readiness.source_sha256)
    host = plan.contract["host"]
    max_active_requests = None
    configured_cap = os.environ.get("NEMOTRON_MAX_ACTIVE_REQUESTS")
    if configured_cap is not None:
        try:
            max_active_requests = json.loads(configured_cap)
        except (ValueError, TypeError):
            raise BundleError("NEMOTRON_MAX_ACTIVE_REQUESTS requires a JSON integer") from None
        if (
            host.get("batching_mode", "fixed_cohort") != "inflight"
            or type(max_active_requests) is not int
            or not 1 <= max_active_requests <= plan.physical_width
        ):
            raise BundleError(
                "NEMOTRON_MAX_ACTIVE_REQUESTS requires inflight mode and integer 1..physical width"
            )
    if (
        host["backend_executor_threads"] != 1
        or host["backend_delivery_group"] != 64
        or host["deferred_audit"]
    ):
        raise BundleError("Changed imported worker ownership or delivery contract")
    configure(
        intraop_threads=host["torch_intra_threads"], interop_threads=host["torch_interop_threads"]
    )
    install_serialization(protocol)

    def engine_factory(width):
        if width == 1:
            with _CONSTRUCTION_LOCK:
                previous_grid = os.environ.get("NEMOTRON_STAGE0_MEGAKERNEL_GRID")
                try:
                    value = readiness.factory()
                finally:
                    if previous_grid is None:
                        os.environ.pop("NEMOTRON_STAGE0_MEGAKERNEL_GRID", None)
                    else:
                        os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = previous_grid
            if value.batch != 1 or value.steps != 511 or value.active_lanes_config:
                value.close()
                raise BundleError("Readiness factory returned an incompatible native engine")
            value.retain_codes = False
            return value
        if width != plan.physical_width:
            raise BundleError("Unregistered physical neural width")
        return create_for_qualification(
            plan,
            artifact_root,
            model_root,
            source_pins=source_pins,
            model_identity=model_identity,
            qualification_only=True,
        )

    if engine_factory_override is not None:
        from .execution.program import ProgramFactory

        if type(engine_factory_override) is not ProgramFactory:
            raise BundleError("Serving requires the source-verified program factory")
        engine_factory_override.verify(plan, readiness)
        engine_factory = engine_factory_override

    manifest = NativeManifest(plan)
    backend = NativeBackend(
        manifest,
        plan.physical_width,
        receipts,
        engine_factory=engine_factory,
        delivery_group=host["backend_delivery_group"],
    )
    try:
        await backend.prepare()
        (backend.receipt_dir / "cpu-pools.json").write_text(
            json.dumps(snapshot(backend, asyncio.get_running_loop()), indent=2) + "\n"
        )
        mode = host.get("batching_mode", "fixed_cohort")
        if mode == "fixed_cohort":
            admission = SimpleNamespace(TimedBatcher=TimedBatcher, BackendError=BackendError)
            if plan.physical_width > 1:
                install_exact_admission(
                    admission,
                    incomplete_timeout_seconds=plan.contract["publication"][
                        "incomplete_deadline_ms"
                    ]
                    / 1000.0,
                )
            batcher = admission.TimedBatcher(
                backend,
                batch_window_ms=0,
                cohort_timeout_ms=1000,
                expected_cohort_size=plan.physical_width,
                max_queue_depth=4 * plan.physical_width,
                max_inflight_batches=2,
                chunk_queue_depth=4,
                cohort_hold=True,
            )
        else:
            from .inflight_batcher import InflightBatcher

            if max_active_requests is None:
                batcher = InflightBatcher(backend, mode=mode)
            else:
                batcher = InflightBatcher(
                    backend, mode=mode, max_active_requests=max_active_requests
                )
                with (backend.receipt_dir / "admission-cap.json").open("x") as stream:
                    json.dump(
                        dict(
                            schema="nemotron.active-admission-cap.binding.v1",
                            status="CONFIGURED_NOT_STARTED",
                            configuration="NEMOTRON_MAX_ACTIVE_REQUESTS",
                            mode=mode,
                            physical_batch=backend.batch,
                            max_active_requests=batcher.max_active_requests,
                            physical_slot_capacity=len(batcher._slots),
                            physical_epoch_capacity=len(batcher._epochs),
                            max_queue_depth=batcher.max_queue_depth,
                            chunk_queue_depth=batcher.chunk_queue_depth,
                            scope="engine_owners_including_paused_and_deferred",
                            excludes_retired_delivery_owners=True,
                            queue_wait_in_request_latency=True,
                            factory="nemotron_native_backend.worker.prepare_for_qualification",
                            backend=f"{type(backend).__module__}.{type(backend).__qualname__}",
                            controller=f"{type(batcher).__module__}.{type(batcher).__qualname__}",
                        ),
                        stream,
                        indent=2,
                    )
                    stream.write("\n")
        server = NativeSlabServer(
            socket_path=Path(socket_path),
            manifest=manifest,
            backend=backend,
            batcher=batcher,
            max_frame_bytes=67108864,
            host_options=host["transport_options"],
            slab_audit_enabled=slab_audit_enabled,
            trace=trace,
        )
        if mode != "fixed_cohort":
            batcher.on_fatal = server.begin_shutdown
        r31.after_prepare()
        (backend.receipt_dir / "r31-options.json").write_text(
            json.dumps(r31.receipt(), indent=2) + "\n"
        )
        return QualificationWorker(backend, batcher, server, readiness.source_sha256)
    except BaseException:
        await backend.shutdown()
        raise
