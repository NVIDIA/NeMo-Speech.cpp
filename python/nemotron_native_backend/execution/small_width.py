from __future__ import annotations

import ctypes as C
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ..bundle import (
    Artifact,
    BundleError,
    NativeBundlePlan,
    _freeze,
    _relative_path,
    _sha256,
    resolve_artifacts,
)
from . import core, helper_graphs
from .engine import _CONSTRUCTION_LOCK, NativeCohortEngine
from .lane_control import update_first_eos
from .native_assets import ModelAssets
from .native_inputs import requests_from_texts
from .source_build import is_source_build, verify_source_build

WIDTHS = (2, 4, 8)
HOOKS = ("predictor_active_mask", "helper_graphs", "controller_queues", "publication_receipt")
GENERATION = dict(cap=511, startup_skip=True, chunk_frames=[3, 5, 8], contraction=False)
STATE = dict(
    format="FP32-private",
    recurrence_arithmetic="FP32",
    canonical_binding=False,
    prefix_cache=False,
    initialization="prefill-per-cohort",
)
PUBLICATION = dict(
    continuation_proof_required=True,
    sample_rate=22050,
    neural_output="pcm_f32_le",
    http_output="pcm_s16_le",
)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_small_width_plan(path: Path) -> NativeBundlePlan:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    width = value.get("physical_width")
    if (
        value.get("schema") != "nemotron.small-width-plan.r22.v1"
        or type(width) is not int
        or width not in WIDTHS
        or value.get("status") != "BUILT_UNQUALIFIED"
        or value.get("enabled") is not False
        or value.get("neural_cohort_partitioning") is not False
    ):
        raise BundleError("Expected a disabled built physical-B2/B4/B8 qualification plan")
    if value.get("generation") != GENERATION or value.get("state") != STATE:
        raise BundleError("Small-width generation or FP32 state contract changed")
    if value.get("publication") != PUBLICATION or tuple(value.get("hooks", ())) != HOOKS:
        raise BundleError("Small-width publication dependencies or hook order changed")
    execution = value["execution"]
    if execution != dict(mode="serial", grid_policy="static", stage0_grid=132, codec_grid=132):
        raise BundleError("This first small-width port supports only proposed serial132/132")
    if value["resources"] != dict(
        occupancy_qualified=False,
        admission_checked_at_construction=True,
        compiled_arch="sm_90a",
        producer_barrier="actual-gridDim.x",
    ):
        raise BundleError("Resource qualification status or barrier contract changed")
    if value["geometry"] != dict(
        physical_requests=width,
        producer_tile_rows=16,
        prefill_tile_rows=64,
        sparse_expert_tile_rows=16,
    ):
        raise BundleError("Physical width and padded arithmetic geometry disagree")
    if value["rng"] != dict(
        groups=1,
        generated_rows_per_draw=64,
        consumed_rows=width,
        unused_rows=64 - width,
        context_draws=1,
        all_physical_generations_advance=True,
        regrouping_equivalence=False,
    ):
        raise BundleError("Small-width independent64 RNG contract changed")
    if value["kernels"] != dict(prefill="prefill", stage0="stage0", codec="codec"):
        raise BundleError("Three explicit neural library bindings are required")
    _sha256(value["model"]["checkpoint_sha256"])
    artifacts = tuple(
        Artifact(name, _relative_path(row["path"]), _sha256(row["sha256"]))
        for name, row in sorted(value["artifacts"].items())
    )
    required = {
        "prefill",
        "stage0",
        "codec",
        "source_manifest",
        "external_source_manifest",
        "prefill_build_receipt",
        "stage0_build_receipt",
        "codec_build_receipt",
    }
    if is_source_build(value):
        required = {"prefill", "stage0", "codec", "source_build_receipt"}
    if not required <= {item.name for item in artifacts}:
        raise BundleError("Missing source, build or library provenance")
    if len({item.relative_path for item in artifacts}) != len(artifacts):
        raise BundleError("Small-width artifact destinations collide")
    return NativeBundlePlan(
        value["id"], width, artifacts, _freeze(value), hashlib.sha256(payload).hexdigest()
    )


def verify_builds(plan, artifacts):
    if is_source_build(plan.contract):
        verify_source_build(plan, artifacts)
        return
    manifest = json.loads(artifacts["source_manifest"].read_text())
    if manifest["batch"] != plan.physical_width:
        raise BundleError("Source manifest physical width differs from the program")
    expected_manifest = _digest(artifacts["source_manifest"])
    expected_external = _digest(artifacts["external_source_manifest"])
    for region in ("prefill", "stage0", "codec"):
        build = json.loads(artifacts[region + "_build_receipt"].read_text())
        if (
            build.get("schema") != "nemotron.small-width-build.r22.v1"
            or build.get("status") != "COMPILED"
            or build.get("compile_returncode") != 0
            or build.get("physical_batch") != plan.physical_width
            or build.get("region") != region
            or build.get("source_manifest_sha256") != expected_manifest
            or build.get("external_source_manifest_sha256") != expected_external
            or build.get("library_sha256") != _digest(artifacts[region])
        ):
            raise BundleError("Unverified or mismatched small-width build: " + region)
        if not build.get("command") or not build.get("image") or not build.get("compiler"):
            raise BundleError("Missing compiler invocation provenance: " + region)
        expected = {
            Path(row["path"]).name: row["sha256"]
            for row in manifest["files"]
            if row["region"] == region
        }
        observed = {Path(row["path"]).name: row["sha256"] for row in build["sources"]}
        if expected != observed or len(observed) != len(build["sources"]):
            raise BundleError("Build source closure differs from prepared sources: " + region)
        for name, digest in expected.items():
            source = artifacts.get(region + "_source_" + name)
            if source is None or _digest(source) != digest:
                raise BundleError(
                    "Packaged build source missing or changed: " + region + "/" + name
                )
        if region == "codec":
            if (
                artifacts["codec_build_receipt"].parent != artifacts["codec"].parent
                or artifacts["codec_build_receipt"].name != "build-receipt.json"
            ):
                raise BundleError("Codec requires its adjacent immutable build receipt")


class SmallWidthSupport:
    ModelAssets = ModelAssets
    Cohort = core.Cohort
    requests_from_texts = staticmethod(requests_from_texts)
    update_first_eos = staticmethod(update_first_eos)

    def __init__(self, plan, artifacts, source_pins):
        self.plan = plan
        self.artifacts = artifacts
        self.source_pins = dict(source_pins)
        self.source_sha256 = _digest(core.__file__)

    def verify_sources(self):
        package = Path(__file__).resolve().parent.parent
        actual = {str(path.relative_to(package)) for path in package.rglob("*.py")}
        if not self.source_pins or actual != set(self.source_pins):
            raise BundleError("Small-width package source inventory differs from its freeze")
        for name, expected in self.source_pins.items():
            path = (package / name).resolve(strict=True)
            if not path.is_relative_to(package) or _digest(path) != expected:
                raise BundleError("Small-width package source changed: " + name)

    def select(self, batch):
        if batch != self.plan.physical_width or batch not in WIDTHS:
            raise BundleError("No small-width fallback or physical cohort partitioning is allowed")
        paths = {
            name: self.artifacts[key]
            for name, key in (("prefill", "prefill"), ("stage0", "stage0"), ("stage1", "codec"))
        }
        return paths, dict(
            physical_batch=batch,
            variant=self.plan.identity,
            neural_cohort_partitioning=False,
            bundle_manifest_sha256=self.plan.manifest_sha256,
        )


def bind_startup_mask(neural):
    function = neural.lib.mk2_set_native_predictor_active_mask
    function.argtypes = [C.c_void_p, C.c_void_p]
    function.restype = C.c_int
    neural._check(
        function(neural.stage0_handle, neural.inputs.audio_valid.data_ptr()), "startup mask"
    )
    neural._startup_skip_mask_owner = neural.inputs.audio_valid


class SmallWidthEngine(NativeCohortEngine):
    def __init__(
        self,
        plan,
        artifacts,
        artifact_root,
        model_root,
        source_pins,
        retain_codes,
        *,
        research_schedule=(3, 5),
        research_steps=None,
    ):
        self.native_plan = plan
        self.native_artifacts = artifacts
        self.small_width_plan = plan
        self.small_width_resources = None
        support = SmallWidthSupport(plan, artifacts, source_pins)
        super().__init__(
            plan.physical_width,
            model_root=model_root,
            steps=511 if research_steps is None else research_steps,
            retain_codes=retain_codes,
            support=support,
            artifact_root=artifact_root,
            qualification_mode=True,
            execution=dict(plan.contract["execution"]),
            active_lanes={"mode": "disabled"},
            research_schedule=research_schedule,
        )

    def _allocate_queues(self):
        n = self.core.neural
        info = self.residency["stage0"]
        expected = dict(
            batch=self.batch,
            batch_pad=16,
            stages=3,
            kchunk=256,
            direct_input=1,
            skip_hidden_snapshot=1,
            threads=256,
            grid=132,
        )
        if any(info[key] != value for key, value in expected.items()):
            raise BundleError(
                "Compiled producer geometry differs from the small-width source contract"
            )
        if info["blocks_per_sm"] < 1 or info["grid"] > info["sms"] * info["blocks_per_sm"]:
            raise BundleError("Producer cooperative launch is not resident")
        if tuple(n.torch.cuda.get_device_capability(0)) != (9, 0):
            raise BundleError("This source variant is compiled for Hopper sm_90a")
        if self.core.codec.batch != self.batch or self.core.codec.state_slots != self.batch:
            raise BundleError("Codec width or state-slot allocation differs from physical requests")
        if self.residency["codec_blocks"] != 132:
            raise BundleError("Codec resource request was clamped")
        if any(
            n.bindings.ssm_slot_stride[i] != 196608 or n.bindings.conv_slot_stride[i] != 10752
            for i in range(14)
        ):
            raise BundleError("FP32 recurrence or convolution state stride changed")
        if (
            n.noise.batch != self.batch
            or len(n.noise.generators) != 1
            or tuple(n.noise.buffer.shape) != (64, 16, 80)
        ):
            raise BundleError(
                "Small physical cohorts must retain one complete independent64 RNG group"
            )
        bind_startup_mask(n)
        self.startup_predictor_skip = True
        self.helper_graphs = helper_graphs.install(n)
        super()._allocate_queues()
        self.small_width_resources = dict(
            admission_checked=True,
            execution_qualified=False,
            stage0=dict(info),
            codec_blocks=self.residency["codec_blocks"],
            prefill_batch=n.info.batch,
            prefill_padded_rows=n.info.padded_prompt_rows,
            prefill_scratch_bytes=n.info.prefill_scratch_bytes,
            state_format="FP32-private",
            producer_barrier_participants=info["grid"],
            precision_verified_by="pinned-source-and-runtime-strides",
        )

    def stream(self, *values, **keywords):
        iterator = super().stream(*values, **keywords)
        try:
            for event in iterator:
                if event["type"] == "receipt":
                    event["receipt"]["small_width"] = dict(
                        plan_sha256=self.small_width_plan.manifest_sha256,
                        physical_batch=self.batch,
                        neural_cohort_partitioning=False,
                        resources=self.small_width_resources,
                        hooks=list(HOOKS),
                        generated_rng_rows=64,
                        consumed_rng_rows=self.batch,
                        unused_rng_rows=64 - self.batch,
                        context_draws=1,
                        regrouping_equivalence=False,
                        publication_boundary="engine float PCM; not HTTP TTFA",
                        qualification_claim=False,
                    )
                yield event
        finally:
            iterator.close()


def create_small_width_engine(
    plan: NativeBundlePlan,
    artifact_root: Path,
    model_root: Path,
    *,
    source_pins: Mapping[str, str],
    model_identity: str,
    qualification_only: bool,
    retain_codes: bool = False,
):
    if (
        qualification_only is not True
        or plan.contract.get("enabled") is not False
        or plan.contract.get("schema") != "nemotron.small-width-plan.r22.v1"
        or plan.physical_width not in WIDTHS
        or type(retain_codes) is not bool
    ):
        raise BundleError("Only explicitly bound small-width qualification is enabled")
    if model_identity != plan.contract["model"]["checkpoint_sha256"]:
        raise BundleError("Caller model identity differs from the frozen plan")
    artifacts = resolve_artifacts(plan, artifact_root)
    verify_builds(plan, artifacts)
    support = SmallWidthSupport(plan, artifacts, source_pins)
    support.verify_sources()
    with _CONSTRUCTION_LOCK:
        return SmallWidthEngine(
            plan, artifacts, artifact_root, model_root, source_pins, retain_codes
        )
