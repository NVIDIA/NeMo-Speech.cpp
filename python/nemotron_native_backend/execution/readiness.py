from __future__ import annotations

import ctypes as C
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..bundle import (
    Artifact,
    BundleError,
    NativeBundlePlan,
    _freeze,
    _relative_path,
    _sha256,
    resolve_artifacts,
)
from . import helper_graphs
from .composition import PackageSupport
from .engine import _CONSTRUCTION_LOCK, NativeCohortEngine
from .source_build import is_source_build, verify_source_build

READINESS_POLICY = MappingProxyType(
    dict(
        mode="serial",
        grid_policy="overlap-only",
        stage0_grid=96,
        full_stage0_grid=96,
        codec_grid=48,
    )
)


def load_readiness_plan(path: Path) -> NativeBundlePlan:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if (
        value.get("schema") != "nemotron.native-readiness-plan.r22.v1"
        or value.get("enabled") is not False
        or value.get("status") != "PROPOSED_UNQUALIFIED"
        or value.get("physical_width") != 1
        or value.get("neural_cohort_partitioning") is not False
    ):
        raise BundleError("Invalid isolated physical-B1 readiness plan")
    if value["execution"] != dict(READINESS_POLICY):
        raise BundleError("Readiness must retain its serial96/48 resource contract")
    if value["generation"] != dict(
        cap=511, startup_skip=True, chunk_frames=[3, 5, 8], contraction=False
    ):
        raise BundleError("Readiness generation contract changed")
    if value["state"] != dict(
        format="FP32-private",
        recurrence_arithmetic="FP32",
        canonical_binding=False,
        prefix_cache=False,
        initialization="prefill-per-cohort",
    ):
        raise BundleError("Readiness must not inherit wide private-state or prefix-cache hooks")
    if value["hooks"] != [
        "predictor_active_mask",
        "helper_graphs",
        "controller_queues",
        "admission_receipt",
    ]:
        raise BundleError("Readiness hook order changed")
    if value["kernels"] != dict(prefill="prefill", stage0="stage0", codec="codec"):
        raise BundleError("Readiness requires three separately bound native libraries")
    _sha256(value["model"]["checkpoint_sha256"])
    artifacts = tuple(
        Artifact(name, _relative_path(row["path"]), _sha256(row["sha256"]))
        for name, row in sorted(value["artifacts"].items())
    )
    if len({item.relative_path for item in artifacts}) != len(artifacts):
        raise BundleError("Readiness artifact destinations collide")
    required = {
        "prefill",
        "stage0",
        "codec",
        "selected_candidate",
        "prefill_build_receipt",
        "stage0_build_receipt",
        "codec_build_receipt",
    }
    if is_source_build(value):
        required = {"prefill", "stage0", "codec", "source_build_receipt"}
    if not required <= {item.name for item in artifacts}:
        raise BundleError("Readiness is missing an immutable binding or build receipt")
    return NativeBundlePlan(
        value["id"], 1, artifacts, _freeze(value), hashlib.sha256(payload).hexdigest()
    )


def verify_readiness_artifacts(plan, artifacts):
    if is_source_build(plan.contract):
        verify_source_build(plan, artifacts)
        return
    selection = json.loads(artifacts["selected_candidate"].read_text())
    if selection["schema"] != "native-serving-candidate.v1" or set(selection["widths"]) != {"1"}:
        raise BundleError("Wrong historical readiness selection")
    row = selection["widths"]["1"]
    if row["startup_predictor_skip"] is not True:
        raise BundleError("Readiness startup mask qualification is missing")
    for region, legacy in (("prefill", "prefill"), ("stage0", "stage0"), ("codec", "stage1")):
        actual = hashlib.sha256(artifacts[region].read_bytes()).hexdigest()
        receipt = json.loads(artifacts[region + "_build_receipt"].read_text())
        if actual != row[legacy]["library"]["sha256"] or actual != receipt["library_sha256"]:
            raise BundleError(
                "Readiness library differs from its pinned selection/build: " + region
            )
        qualification = artifacts[region + "_qualification"]
        if (
            hashlib.sha256(qualification.read_bytes()).hexdigest()
            != row[legacy]["qualification"]["sha256"]
        ):
            raise BundleError("Readiness qualification source changed: " + region)


def bind_readiness_mask(neural):
    function = neural.lib.mk2_set_native_predictor_active_mask
    function.argtypes = [C.c_void_p, C.c_void_p]
    function.restype = C.c_int
    neural._check(
        function(neural.stage0_handle, neural.inputs.audio_valid.data_ptr()), "startup mask"
    )
    neural._startup_skip_mask_owner = neural.inputs.audio_valid


def bind_admission_receipt(engine, admission):
    original = engine.stream
    frozen = dict(admission)

    def stream(*values, **keywords):
        iterator = original(*values, **keywords)
        try:
            for event in iterator:
                if event["type"] == "receipt":
                    event["receipt"]["r21_admission"] = dict(frozen)
                yield event
        finally:
            iterator.close()

    engine.stream = stream


class ConfiguredReadinessEngine(NativeCohortEngine):
    def __init__(
        self,
        plan,
        artifacts,
        artifact_root,
        model_root,
        source_pins,
        admission,
        *,
        research_schedule=(3, 5),
        research_steps=None,
    ):
        self.native_plan = plan
        self.native_artifacts = artifacts
        self.readiness_plan = plan
        support = PackageSupport(plan, artifacts, source_pins)
        super().__init__(
            1,
            model_root=model_root,
            steps=511 if research_steps is None else research_steps,
            retain_codes=False,
            support=support,
            artifact_root=artifact_root,
            qualification_mode=True,
            execution=dict(plan.contract["execution"]),
            active_lanes={"mode": "disabled"},
            research_schedule=research_schedule,
        )
        bind_admission_receipt(self, admission)

    def _allocate_queues(self):
        bind_readiness_mask(self.core.neural)
        self.startup_predictor_skip = True
        self.helper_graphs = helper_graphs.install(self.core.neural)
        super()._allocate_queues()


@dataclass(frozen=True)
class ReadinessFactory:
    plan: NativeBundlePlan
    plan_path: Path
    artifact_root: Path
    model_root: Path
    source_pins: Mapping[str, str]
    model_identity: str
    admission: Mapping[str, object]
    source_sha256: str

    def verify_identity(self, expected_source_sha256):
        source = Path(__file__).resolve()
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if (
            expected_source_sha256 != self.source_sha256
            or actual != self.source_sha256
            or self.source_pins.get("execution/readiness.py") != actual
        ):
            raise BundleError("Readiness factory source is not the frozen package source")
        if hashlib.sha256(self.plan_path.read_bytes()).hexdigest() != self.plan.manifest_sha256:
            raise BundleError("Readiness manifest changed after binding")
        if self.model_identity != self.plan.contract["model"]["checkpoint_sha256"]:
            raise BundleError("Readiness model identity differs from the wide model")
        artifacts = resolve_artifacts(self.plan, self.artifact_root)
        verify_readiness_artifacts(self.plan, artifacts)
        PackageSupport(self.plan, artifacts, self.source_pins).verify_sources()
        return artifacts

    def __call__(self):
        with _CONSTRUCTION_LOCK:
            artifacts = self.verify_identity(self.source_sha256)
            return ConfiguredReadinessEngine(
                self.plan,
                artifacts,
                self.artifact_root,
                self.model_root,
                self.source_pins,
                self.admission,
            )


def make_readiness_binding(
    path: Path,
    artifact_root: Path,
    model_root: Path,
    *,
    source_pins: Mapping[str, str],
    wide_plan: NativeBundlePlan,
    model_identity: str,
    qualification_only: bool,
):
    if qualification_only is not True or wide_plan.contract["enabled"] is not False:
        raise BundleError("The isolated readiness port is qualification-only")
    plan_path = Path(path).resolve(strict=True)
    plan = load_readiness_plan(plan_path)
    if (
        plan.contract["model"]["checkpoint_sha256"]
        != wide_plan.contract["model"]["checkpoint_sha256"]
    ):
        raise BundleError("Readiness and wide engine use different checkpoints")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    admission = dict(
        initial_singleton_readiness=True,
        post_readiness_width=wide_plan.physical_width,
        incomplete_deadline_ms=wide_plan.contract["publication"]["incomplete_deadline_ms"],
    )
    factory = ReadinessFactory(
        plan,
        plan_path,
        Path(artifact_root).resolve(strict=True),
        Path(model_root).resolve(strict=True),
        MappingProxyType(dict(source_pins)),
        model_identity,
        MappingProxyType(admission),
        source_sha,
    )
    factory.verify_identity(source_sha)
    from ..worker import ReadinessBinding

    return ReadinessBinding(factory=factory, source_sha256=source_sha, physical_width=1)
