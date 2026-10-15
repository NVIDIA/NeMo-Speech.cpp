import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..bundle import BundleError, _freeze, resolve_artifacts

SUPPORTED = {
    "readiness": (1,),
    "small": (2, 4, 8),
    "narrow": (16, 32, 64),
    "wide": (128, 256, 512, 1024, 2048, 4096),
}
ALLOWED = ((3, 5), (2, 3, 4, 5))


@dataclass(frozen=True)
class ResearchProgram:
    identity: str
    schedule: tuple[int, ...]
    horizon: int
    parent_sha256: str
    manifest_sha256: str
    physical_width: int
    family: str


def load_research_program(path, base_plan):
    payload = Path(path).read_bytes()
    v = json.loads(payload)
    family = v.get("family")
    schedule = tuple(v.get("startup", ()))
    if (
        v.get("schema") != "nemotron.chunk-schedule.program.r30.v1"
        or family not in SUPPORTED
        or v.get("physical_width") not in SUPPORTED[family]
        or v.get("physical_width") != base_plan.physical_width
        or schedule not in ALLOWED
        or v.get("steady") != 8
        or v.get("speech_delay") != 5
        or v.get("horizon") not in (28, 511)
        or v.get("parent_generation_cap") != 511
        or v.get("parent_manifest_sha256") != base_plan.manifest_sha256
        or v.get("qualification_only") is not True
        or v.get("serving_enabled") is not False
        or v.get("continuation_proof") is not True
        or v.get("neural_dso_changes") is not False
        or _freeze(v.get("parent_execution")) != base_plan.contract["execution"]
        or _freeze(v.get("parent_state")) != base_plan.contract["state"]
    ):
        raise BundleError("Research width, family, schedule or selected parent contract mismatch")
    return ResearchProgram(
        v["id"],
        schedule,
        v["horizon"],
        base_plan.manifest_sha256,
        hashlib.sha256(payload).hexdigest(),
        base_plan.physical_width,
        family,
    )


def create_research(program, base_plan, artifact_root, model_root, *, source_pins, model_identity):
    from .composition import ConfiguredEngine, PackageSupport, verify_compiled_contract
    from .engine import _CONSTRUCTION_LOCK

    if (
        program.parent_sha256 != base_plan.manifest_sha256
        or program.physical_width != base_plan.physical_width
        or model_identity != base_plan.contract["model"]["checkpoint_sha256"]
    ):
        raise BundleError("Research/base/model identity mismatch")
    artifacts = resolve_artifacts(base_plan, artifact_root)
    PackageSupport(base_plan, artifacts, source_pins).verify_sources()
    paths = (base_plan, artifacts, Path(artifact_root), Path(model_root), source_pins)
    options = dict(research_schedule=program.schedule, research_steps=program.horizon)
    with _CONSTRUCTION_LOCK:
        if program.family == "readiness":
            from .readiness import ConfiguredReadinessEngine, verify_readiness_artifacts

            verify_readiness_artifacts(base_plan, artifacts)
            engine = ConfiguredReadinessEngine(
                *paths,
                dict(
                    initial_singleton_readiness=True,
                    post_readiness_width=1,
                    incomplete_deadline_ms=10000,
                ),
                **options,
            )
        elif program.family == "small":
            from .small_width import SmallWidthEngine, verify_builds

            verify_builds(base_plan, artifacts)
            engine = SmallWidthEngine(*paths, False, **options)
        elif program.family == "narrow":
            from .retained_narrow import ConfiguredNarrowEngine, verify

            engine = ConfiguredNarrowEngine(*paths, verify(base_plan, artifacts), **options)
        elif program.family == "wide":
            engine = ConfiguredEngine(
                *paths, verify_compiled_contract(base_plan, artifacts), **options
            )
        else:
            raise BundleError("Unregistered research family")
    original = engine.stream

    def stream(*args, **kwargs):
        iterator = original(*args, **kwargs)
        try:
            for event in iterator:
                if event["type"] == "receipt":
                    event["receipt"]["research_program"] = dict(
                        id=program.identity,
                        manifest_sha256=program.manifest_sha256,
                        startup=list(program.schedule),
                        steady=8,
                        speech_delay=5,
                        horizon=program.horizon,
                        parent_cap=511,
                        physical_width=program.physical_width,
                        family=program.family,
                        serving_qualified=False,
                        authoritative_HTTP_measurement=False,
                    )
                yield event
        finally:
            iterator.close()

    engine.stream = stream
    return engine
