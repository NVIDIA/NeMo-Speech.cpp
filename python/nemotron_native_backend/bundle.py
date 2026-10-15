from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Artifact:
    name: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class NativeBundlePlan:
    identity: str
    physical_width: int
    artifacts: tuple[Artifact, ...]
    contract: Mapping[str, Any]
    manifest_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _relative_path(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "Artifact path must be nonempty")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute() and ".." not in path.parts, "Artifacts require bundle-relative paths"
    )
    _require(
        "\\" not in value and "\x00" not in value and path.as_posix() == value,
        "Artifact path must be canonical POSIX syntax",
    )
    return value


def _sha256(value: Any) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        "Expected lowercase SHA-256",
    )
    return value


def load_plan(path: Path) -> NativeBundlePlan:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    _require(isinstance(value, dict), "Bundle must be an object")
    _require(value.get("schema") == "nemotron.native-package-plan.r22.v1", "Unknown schema")
    _require(
        value.get("status") == "PROPOSED_UNQUALIFIED" and value.get("enabled") is False,
        "This preparatory loader cannot enable serving",
    )
    width = value["physical_width"]
    _require(type(width) is int and width > 0, "Physical width must be positive")
    _require(value["neural_cohort_partitioning"] is False, "Physical cohorts cannot split")
    artifact_rows = value["artifacts"]
    artifacts = tuple(
        Artifact(name, _relative_path(row["path"]), _sha256(row["sha256"]))
        for name, row in sorted(artifact_rows.items())
    )
    names = {item.name for item in artifacts}
    _require(
        len({item.relative_path for item in artifacts}) == len(artifacts),
        "Artifact destinations must be unique",
    )
    kernels = value["kernels"]
    _require(set(kernels) == {"prefill", "stage0", "codec"}, "Three neural bindings required")
    for region, kernel in kernels.items():
        _require(kernel["artifact"] in names, "Missing kernel artifact: " + region)
        _require(kernel["physical_width"] == width, "Kernel physical widths must agree")
        _require(
            isinstance(kernel["entrypoint"], str) and kernel["entrypoint"].isidentifier(),
            "Invalid kernel entrypoint",
        )
    generation = value["generation"]
    _require(
        generation["cap"] == 511
        and generation["startup_skip"] is True
        and generation["chunk_frames"] == [3, 5, 8],
        "Unsupported generation contract",
    )
    _require(generation["contraction"] is False, "This imported wide plan has no contraction")
    publication = value["publication"]
    _require(
        publication["continuation_proof_required"] is True
        and publication["sample_rate"] == 22050
        and publication["neural_output"] == "pcm_f32_le"
        and publication["http_output"] == "pcm_s16_le",
        "Changed publication contract",
    )
    state = value["state"]
    _require(
        state["mode"] == 1
        and state["format"] == "canonical-FP32-private-BF16"
        and state["recurrence_arithmetic"] == "FP32",
        "Unsupported imported state format",
    )
    _require(
        state["scales_artifact"] in names and state["map_artifact"] in names,
        "State bindings require explicit scale and map artifacts",
    )
    execution = value["execution"]
    _require(
        execution["mode"] == "serial" and execution["grid_policy"] == "overlap-only",
        "Imported plan is qualified only for serial execution",
    )
    for key in ("stage0_grid", "full_stage0_grid", "codec_grid"):
        _require(type(execution[key]) is int and execution[key] > 0, "Invalid resource count")
    _require(
        execution["stage0_grid"] == execution["full_stage0_grid"],
        "Serial producer allocation differs from compiled participant count",
    )
    _require(
        kernels["stage0"]["barrier_participants"] == execution["stage0_grid"],
        "Producer barrier participants differ from launch",
    )
    _sha256(value["model"]["checkpoint_sha256"])
    host = value["host"]
    _require(
        host["tokens"] == "pinned" and host["serialization"] == "msgpack-c-1.2.2",
        "Unrecognized imported host path",
    )
    _require(
        host["pcm_conversion"] == "exact-avx2-with-scalar-fallback",
        "SIMD changes require a separately qualified plan",
    )
    _require(
        value["transport_compatibility"]["identity_migration_enabled"] is False,
        "Legacy ready identity must remain compatible until qualification",
    )
    return NativeBundlePlan(
        value["id"], width, artifacts, _freeze(value), hashlib.sha256(payload).hexdigest()
    )


def resolve_artifacts(plan: NativeBundlePlan, artifact_root: Path) -> Mapping[str, Path]:
    root = Path(artifact_root).resolve(strict=True)
    resolved = {}
    for artifact in plan.artifacts:
        path = (root / artifact.relative_path).resolve(strict=True)
        _require(path.is_relative_to(root) and path.is_file(), "Artifact escapes bundle root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        _require(digest.hexdigest() == artifact.sha256, "Artifact hash mismatch: " + artifact.name)
        resolved[artifact.name] = path
    return MappingProxyType(resolved)


def create_engine(
    plan: NativeBundlePlan,
    artifact_root: Path,
    model_root: Path,
    *,
    source_pins: Mapping[str, str] | None = None,
    model_identity: str | None = None,
    qualification_only: bool = False,
) -> Any:
    if qualification_only is not True or source_pins is None or model_identity is None:
        raise BundleError(
            "Production serving remains disabled; explicit qualification bindings are required"
        )
    from .execution.composition import create_for_qualification

    return create_for_qualification(
        plan,
        artifact_root,
        model_root,
        source_pins=source_pins,
        model_identity=model_identity,
        qualification_only=True,
    )
