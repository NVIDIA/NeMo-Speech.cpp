from __future__ import annotations

import hashlib
import json
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
from . import helper_graphs, narrow_prefix_cache, narrow_state_binding
from .composition import PackageSupport
from .engine import _CONSTRUCTION_LOCK, NativeCohortEngine
from .readiness import bind_readiness_mask
from .source_build import is_source_build, verify_source_build


def load_plan(path):
    payload = Path(path).read_bytes()
    v = json.loads(payload)
    width = v.get("physical_width")
    if (
        v.get("schema") != "nemotron.retained-narrow.r22.v1"
        or width not in (16, 32, 64)
        or v.get("enabled") is not False
        or v.get("status") != "PROPOSED_UNQUALIFIED"
        or v.get("neural_cohort_partitioning") is not False
    ):
        raise BundleError("Unsupported retained narrow contract")
    execution = dict(
        mode="overlap",
        grid_policy="overlap-only",
        stage0_grid=80,
        full_stage0_grid=132,
        codec_grid=48 if width == 64 else 52,
    )
    if v["execution"] != execution:
        raise BundleError("Retained resource policy changed")
    if v["generation"] != dict(cap=511, startup_skip=True, chunk_frames=[3, 5, 8]):
        raise BundleError("Retained generation contract changed")
    if v["state"] != dict(
        format="BF16-private" if width == 32 else "FP32-private",
        recurrence="FP32",
        prefix_cache=width == 32,
    ):
        raise BundleError("Retained state/cache contract changed")
    if v["predictor"] != dict(full_width="FP8-FFN" if width == 32 else "BF16", contracted="BF16"):
        raise BundleError("Retained predictor precision changed")
    if width == 16 and v["active_lanes"] != {"mode": "disabled"}:
        raise BundleError("B16 must not contract")
    if width != 16:
        active = v["active_lanes"]
        if any(
            active.get(k) != x
            for k, x in dict(
                mode="fixed-subset",
                target_batch=16,
                after_chunk0=True,
                narrow_grid=80,
                grid_policy="follow-stage0",
            ).items()
        ):
            raise BundleError("Retained contraction policy changed")
    artifacts = tuple(
        Artifact(name, _relative_path(row["path"]), _sha256(row["sha256"]))
        for name, row in sorted(v["artifacts"].items())
    )
    if len({a.relative_path for a in artifacts}) != len(artifacts):
        raise BundleError("Artifact destinations collide")
    required = {
        "prefill",
        "stage0",
        "codec",
        "selected_candidate",
        "compiled_binding",
        "retained_runtime",
    }
    if is_source_build(v):
        required = {"prefill", "stage0", "codec", "source_build_receipt"}
        if width == 32:
            required.add("contraction_bf16")
    if not required <= {a.name for a in artifacts}:
        raise BundleError("Missing retained binding evidence")
    return NativeBundlePlan(
        v["id"], width, artifacts, _freeze(v), hashlib.sha256(payload).hexdigest()
    )


def verify(plan, artifacts):
    if is_source_build(plan.contract):
        receipt = verify_source_build(plan, artifacts)
        if not receipt.get("expected_residency"):
            raise BundleError("Fresh narrow build requires an explicit resource contract")
        return receipt["expected_residency"]
    binding = json.loads(artifacts["compiled_binding"].read_text())
    if (
        binding["physical_batch"] != plan.physical_width
        or binding["neural_cohort_partitioning"] is not False
    ):
        raise BundleError("Actual compiled width differs from plan")
    selection = json.loads(artifacts["selected_candidate"].read_text())["widths"][
        str(plan.physical_width)
    ]
    config = (
        json.loads(artifacts["numerical_config"].read_text()) if plan.physical_width == 32 else None
    )
    for region in ("prefill", "stage0", "codec"):
        legacy = "stage1" if region == "codec" else region
        actual = hashlib.sha256(artifacts[region].read_bytes()).hexdigest()
        expected = (
            config["components"][legacy]["sha256"]
            if config
            else selection[legacy]["library"]["sha256"]
        )
        receipt = json.loads(artifacts[region + "_build_receipt"].read_text())
        if actual != expected or receipt["library_sha256"] != actual:
            raise BundleError("Retained library identity mismatch: " + region)
    if config:
        expected = config["hooks"][0]["kwargs"]["narrow_sha256"]
        if hashlib.sha256(artifacts["contraction_bf16"].read_bytes()).hexdigest() != expected:
            raise BundleError("B32 contraction precision artifact mismatch")
    return json.loads(artifacts["retained_runtime"].read_text())["native"]["residency"]


class ConfiguredNarrowEngine(NativeCohortEngine):
    def __init__(
        self,
        plan,
        artifacts,
        artifact_root,
        model_root,
        source_pins,
        expected,
        *,
        research_schedule=(3, 5),
        research_steps=None,
    ):
        self.native_plan = plan
        self.native_artifacts = artifacts
        active = {
            k: dict(v) if hasattr(v, "items") else v
            for k, v in plan.contract["active_lanes"].items()
        }
        super().__init__(
            plan.physical_width,
            model_root=model_root,
            steps=511 if research_steps is None else research_steps,
            retain_codes=False,
            support=PackageSupport(plan, artifacts, source_pins),
            artifact_root=artifact_root,
            qualification_mode=True,
            execution=dict(plan.contract["execution"]),
            active_lanes=active,
            research_schedule=research_schedule,
        )
        try:
            actual = self.residency
            for name, value in expected["stage0"].items():
                if name != "device" and actual["stage0"][name] != value:
                    raise BundleError("Actual producer ABI differs from retained contract: " + name)
            if actual["codec_blocks"] != expected["codec_blocks"]:
                raise BundleError("Actual codec resource binding changed")
            if self.batch == 32:
                narrow = artifacts["contraction_bf16"]
                narrow_state_binding.install(
                    self, narrow, hashlib.sha256(narrow.read_bytes()).hexdigest(), artifact_root
                )
                narrow_prefix_cache.install(self)
            self._narrow_retained_recipe = dict(plan.contract["state"])
        except BaseException:
            self.close()
            raise

    def _allocate_queues(self):
        bind_readiness_mask(self.core.neural)
        self.startup_predictor_skip = True
        self.helper_graphs = helper_graphs.install(self.core.neural)
        super()._allocate_queues()


def create_for_qualification(
    plan, artifact_root, model_root, *, source_pins, model_identity, qualification_only
):
    if qualification_only is not True or plan.contract["enabled"] is not False:
        raise BundleError("Narrow port is qualification-only")
    if model_identity != plan.contract["model"]["checkpoint_sha256"]:
        raise BundleError("Model identity mismatch")
    artifacts = resolve_artifacts(plan, artifact_root)
    expected = verify(plan, artifacts)
    with _CONSTRUCTION_LOCK:
        return ConfiguredNarrowEngine(
            plan, artifacts, Path(artifact_root), Path(model_root), source_pins, expected
        )
