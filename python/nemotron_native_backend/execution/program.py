from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..bundle import BundleError, NativeBundlePlan, _freeze
from ..bundle import load_plan as load_wide_plan
from .composition import PackageSupport
from .composition import create_for_qualification as create_wide
from .engine import _CONSTRUCTION_LOCK
from .readiness import load_readiness_plan
from .retained_narrow import create_for_qualification as create_narrow
from .retained_narrow import load_plan as load_narrow_plan
from .small_width import create_small_width_engine, load_small_width_plan

FAMILIES = {
    "readiness": (1,),
    "small": (2, 4, 8),
    "narrow": (16, 32, 64),
    "wide": (128, 256, 512, 1024, 2048, 4096),
}


def load_program(path, family):
    loaders = {
        "readiness": load_readiness_plan,
        "small": load_small_width_plan,
        "narrow": load_narrow_plan,
        "wide": load_wide_plan,
    }
    if family not in loaders:
        raise BundleError("Unknown native program family")
    plan = loaders[family](path)
    if plan.physical_width not in FAMILIES[family]:
        raise BundleError("Neural width does not belong to its declared family")
    return plan


def load_serving_plan(path, neural, family):
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if (
        value.get("schema") != "nemotron.unified-serving.r22.v1"
        or value.get("enabled") is not False
        or value.get("family") != family
        or value.get("physical_width") != neural.physical_width
        or value.get("neural_manifest_sha256") != neural.manifest_sha256
        or value["model"]["checkpoint_sha256"] != neural.contract["model"]["checkpoint_sha256"]
    ):
        raise BundleError("Serving and neural programs disagree")
    policy = value["host"].get("fp32_prefix_snapshot", "disabled")
    batching = value["host"].get("batching_mode", "fixed_cohort")
    request_seed = value["host"].get("request_seed", 0)
    if type(request_seed) is not int or not 0 <= request_seed <= 0xFFFFFFFFFFFFFFFF:
        raise BundleError("Request seed must fit uint64")
    if batching not in ("fixed_cohort", "fixed_request", "inflight"):
        raise BundleError("Unsupported request batching mode or physical width")
    if batching != "fixed_cohort" and "stage0_inflight" not in {a.name for a in neural.artifacts}:
        raise BundleError("Request batching requires a source-bound masked producer")
    if "inflight_policy" in value["host"]:
        from .inflight_policy import InflightPolicy

        try:
            InflightPolicy.parse(value["host"]["inflight_policy"])
        except ValueError as error:
            raise BundleError(str(error)) from error
        if batching == "fixed_cohort":
            raise BundleError("In-flight policy cannot alter the fixed-cohort path")
    if policy not in ("disabled", "immutable-fp32-v3") or (
        policy != "disabled" and neural.physical_width not in (1, 2, 4, 8, 16, 64)
    ):
        raise BundleError("Unsupported FP32 prefix policy or physical width")
    research = value.get("research_chunk_schedule")
    if research is not None:
        from .research_program import load_research_program

        research_path = Path(path).with_name("research.json")
        program = load_research_program(research_path, neural)
        if (
            research
            != {
                "program_sha256": program.manifest_sha256,
                "startup": list(program.schedule),
                "steady": 8,
                "horizon": 511,
                "qualification_only": True,
            }
            or program.horizon != 511
        ):
            raise BundleError("Changed research serving schedule or horizon")
    for section in ("generation", "execution", "state"):
        expected = neural.contract[section]
        if section == "generation" and research is not None:
            expected = dict(expected, chunk_frames=tuple(program.schedule) + (8,))
        if section in value and _freeze(value[section]) != _freeze(expected):
            raise BundleError("Serving wrapper changes the neural contract: " + section)
    return NativeBundlePlan(
        value["id"],
        neural.physical_width,
        neural.artifacts,
        _freeze(value),
        hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True)
class ProgramFactory:
    family: str
    neural: NativeBundlePlan
    neural_path: Path
    artifact_root: Path
    model_root: Path
    source_pins: Mapping[str, str]
    model_identity: str
    readiness: object
    serving_sha256: str
    token_policy: str
    prefix_snapshot_policy: str = "disabled"
    research_path: Path | None = None
    research_sha256: str | None = None

    def verify(self, serving, readiness):
        if (
            self.family not in FAMILIES
            or self.neural.physical_width not in FAMILIES[self.family]
            or serving.physical_width != self.neural.physical_width
            or serving.manifest_sha256 != self.serving_sha256
            or serving.contract["neural_manifest_sha256"] != self.neural.manifest_sha256
            or serving.contract["host"]["tokens"] != self.token_policy
            or self.token_policy not in ("pinned", "retained-engine-token-packing")
            or serving.contract["host"].get("fp32_prefix_snapshot", "disabled")
            != self.prefix_snapshot_policy
            or self.prefix_snapshot_policy not in ("disabled", "immutable-fp32-v3")
            or (
                self.prefix_snapshot_policy != "disabled"
                and self.neural.physical_width not in (1, 2, 4, 8, 16, 64)
            )
            or readiness is not self.readiness
            or self.model_identity != self.neural.contract["model"]["checkpoint_sha256"]
        ):
            raise BundleError("Invalid serving program binding")
        research = serving.contract.get("research_chunk_schedule")
        if research is None:
            if self.research_path is not None or self.research_sha256 is not None:
                raise BundleError("Unexpected research factory")
        elif (
            self.research_path is None
            or research["program_sha256"] != self.research_sha256
            or research["horizon"] != 511
            or research["qualification_only"] is not True
        ):
            raise BundleError("Research factory does not match serving manifest")
        self._verify_sources()

    def _verify_sources(self):
        if hashlib.sha256(self.neural_path.read_bytes()).hexdigest() != self.neural.manifest_sha256:
            raise BundleError("Neural program changed after binding")
        PackageSupport(self.neural, {}, self.source_pins).verify_sources()
        self.readiness.factory.verify_identity(self.readiness.source_sha256)
        if self.research_path is not None:
            from .research_program import load_research_program

            program = load_research_program(self.research_path, self.neural)
            if program.manifest_sha256 != self.research_sha256 or program.horizon != 511:
                raise BundleError("Research program changed after binding")

    def __call__(self, width):
        if self.prefix_snapshot_policy not in ("disabled", "immutable-fp32-v3"):
            raise BundleError("Unsupported FP32 prefix policy")
        if width not in (1, self.neural.physical_width):
            raise BundleError("Unregistered physical width")
        with _CONSTRUCTION_LOCK:
            self._verify_sources()
            previous = os.environ.get("NEMOTRON_STAGE0_MEGAKERNEL_GRID")
            try:
                if width == 1 and (self.neural.physical_width != 1 or self.research_path is None):
                    engine = self.readiness.factory()
                elif self.research_path is not None:
                    from .research_program import create_research, load_research_program

                    program = load_research_program(self.research_path, self.neural)
                    engine = create_research(
                        program,
                        self.neural,
                        self.artifact_root,
                        self.model_root,
                        source_pins=self.source_pins,
                        model_identity=self.model_identity,
                    )
                else:
                    constructors = {
                        "small": create_small_width_engine,
                        "narrow": create_narrow,
                        "wide": create_wide,
                    }
                    engine = constructors[self.family](
                        self.neural,
                        self.artifact_root,
                        self.model_root,
                        source_pins=self.source_pins,
                        model_identity=self.model_identity,
                        qualification_only=True,
                    )
                if engine.batch != width or engine.steps != 511:
                    engine.close()
                    raise BundleError("Constructed native engine differs from its program")
                if width == 1 and engine.active_lanes_config:
                    engine.close()
                    raise BundleError("Readiness cannot inherit contraction")
                if width == self.neural.physical_width and self.token_policy == "pinned":
                    from .token_staging import install

                    install(engine, mode="pinned")
                if (
                    width == self.neural.physical_width
                    and self.prefix_snapshot_policy == "immutable-fp32-v3"
                ):
                    from .prefix_snapshot import install as install_prefix

                    try:
                        install_prefix(engine)
                    except BaseException:
                        engine.close()
                        raise
                engine.retain_codes = False
                return engine
            finally:
                if previous is None:
                    os.environ.pop("NEMOTRON_STAGE0_MEGAKERNEL_GRID", None)
                else:
                    os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = previous


async def prepare_for_qualification(
    plan,
    neural,
    neural_path,
    artifact_root,
    model_root,
    *,
    source_pins,
    model_identity,
    readiness,
    socket_path,
    receipts,
    qualification_only,
    slab_audit_enabled=False,
    trace=None,
):
    from ..worker import prepare_for_qualification as prepare_worker

    factory = ProgramFactory(
        plan.contract["family"],
        neural,
        Path(neural_path).resolve(strict=True),
        Path(artifact_root).resolve(strict=True),
        Path(model_root).resolve(strict=True),
        _freeze(dict(source_pins)),
        model_identity,
        readiness,
        plan.manifest_sha256,
        plan.contract["host"]["tokens"],
        plan.contract["host"].get("fp32_prefix_snapshot", "disabled"),
        Path(neural_path).with_name("research.json")
        if "research_chunk_schedule" in plan.contract
        else None,
        plan.contract.get("research_chunk_schedule", {}).get("program_sha256"),
    )
    return await prepare_worker(
        plan,
        artifact_root,
        model_root,
        source_pins=source_pins,
        model_identity=model_identity,
        readiness=readiness,
        socket_path=socket_path,
        receipts=receipts,
        qualification_only=qualification_only,
        slab_audit_enabled=slab_audit_enabled,
        trace=trace,
        engine_factory_override=factory,
    )
