from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..bundle import BundleError, _relative_path, _sha256

SCHEMA = "nemotron.native-source-build.v1"
LIBRARY_SCHEMA = "nemotron.native-source-library.v1"


def is_source_build(contract):
    return type(contract.get("source_build")) is int and contract["source_build"] == 1


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def _resolve(root, value):
    path = (root / _relative_path(value)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise BundleError("Source-build artifact escapes its package")
    return path


def _verify_receipt(path, width):
    path = Path(path).resolve(strict=True)
    root = path.parent
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "COMPILED"
        or type(receipt.get("physical_width")) is not int
        or receipt["physical_width"] != width
    ):
        raise BundleError("Invalid unqualified source-build receipt")
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise BundleError("Source build requires its complete source closure")
    seen = set()
    for row in sources:
        relative = _relative_path(row["path"])
        if not relative.startswith("sources/") or relative in seen:
            raise BundleError("Source closure contains duplicate or non-source entries")
        seen.add(relative)
        source = _resolve(root, relative)
        if _digest(source) != _sha256(row["sha256"]):
            raise BundleError("Source-build source changed: " + relative)
    libraries = receipt.get("libraries")
    if not isinstance(libraries, dict) or not {"prefill", "stage0", "codec"} <= set(libraries):
        raise BundleError("Source build requires three neural library bindings")
    destinations = set()
    for role, row in libraries.items():
        library = _resolve(root, row["path"])
        if library in destinations:
            raise BundleError("Source-build library destinations collide")
        destinations.add(library)
        digest = _digest(library)
        if digest != _sha256(row["sha256"]):
            raise BundleError("Source-build library changed: " + role)
        build_path = _resolve(root, row["receipt"])
        if _digest(build_path) != _sha256(row["receipt_sha256"]):
            raise BundleError("Source-build compiler receipt changed: " + role)
        build = json.loads(build_path.read_text())
        if (
            build.get("schema") != LIBRARY_SCHEMA
            or build.get("status") != "COMPILED"
            or build.get("compile_returncode") != 0
            or build.get("physical_width") != width
            or build.get("region") != role
            or build.get("library_sha256") != digest
            or not build.get("command")
            or not build.get("compiler")
        ):
            raise BundleError("Invalid source-build compiler receipt: " + role)
        pointer = (build_path.parent / build["source_build_receipt"]).resolve(strict=True)
        if pointer != path:
            raise BundleError("Compiler receipt refers to another source package")
    return receipt


def verify_source_build(plan, artifacts):
    if (
        not is_source_build(plan.contract)
        or plan.contract.get("enabled") is not False
        or plan.contract.get("status") not in {"PROPOSED_UNQUALIFIED", "BUILT_UNQUALIFIED"}
    ):
        raise BundleError("Fresh source builds are qualification-only")
    source_receipt = artifacts.get("source_build_receipt")
    if source_receipt is None:
        raise BundleError("Missing source-build receipt")
    source_receipt = Path(source_receipt).resolve(strict=True)
    receipt = _verify_receipt(source_receipt, plan.physical_width)
    root = source_receipt.parent
    for role, row in receipt["libraries"].items():
        if role not in artifacts or Path(artifacts[role]).resolve() != _resolve(root, row["path"]):
            raise BundleError("Plan and source-build library binding disagree: " + role)
    shared_objects = {name for name, path in artifacts.items() if Path(path).suffix == ".so"}
    if shared_objects != set(receipt["libraries"]):
        raise BundleError("Source build omits a plan library")
    active = plan.contract.get("active_lanes", {})
    if active.get("mode") == "fixed-subset":
        known = {
            (_resolve(root, row["path"]), row["sha256"]) for row in receipt["libraries"].values()
        }
        for role in ("stage0", "transfer"):
            row = active[role]
            if (_resolve(root, row["path"]), row["sha256"]) not in known:
                raise BundleError("Contraction library is outside the source build")
    expected = receipt.get("expected_residency")
    if expected is not None:
        producer = expected.get("stage0", {})
        execution = plan.contract["execution"]
        if (
            producer.get("batch") != plan.physical_width
            or producer.get("grid") != execution.get("full_stage0_grid", execution["stage0_grid"])
            or expected.get("codec_blocks") != execution["codec_grid"]
        ):
            raise BundleError("Source-build resources differ from the plan")
    return receipt


def source_library_receipt(library, width):
    library = Path(library).resolve(strict=True)
    build_path = library.parent / "build-receipt.json"
    build = json.loads(build_path.read_text())
    if build.get("schema") != LIBRARY_SCHEMA:
        return None
    pointer = build.get("source_build_receipt")
    if not isinstance(pointer, str) or Path(pointer).is_absolute():
        raise BundleError("Invalid source-build receipt pointer")
    source_path = (build_path.parent / pointer).resolve(strict=True)
    if not library.is_relative_to(source_path.parent):
        raise BundleError("Source-build receipt is outside the library package")
    receipt = _verify_receipt(source_path, width)
    codec = receipt["libraries"]["codec"]
    if _resolve(source_path.parent, codec["path"]) != library:
        raise BundleError("Codec does not match its source-build package")
    return dict(
        build_receipt=str(build_path),
        build_receipt_sha256=_digest(build_path),
        source_build_receipt=str(source_path),
        source_build_receipt_sha256=_digest(source_path),
        source_closure=[
            {"path": str(_resolve(source_path.parent, row["path"])), "sha256": row["sha256"]}
            for row in receipt["sources"]
        ],
    )
