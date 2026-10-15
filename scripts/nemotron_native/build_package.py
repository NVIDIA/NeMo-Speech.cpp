import argparse
import copy
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

from build_kernels import (
    REPOSITORY,
    SOURCE_ROOT,
    SUPPORTED_WIDTHS,
    confined,
    digest,
    load_manifest,
    parse_widths,
    write_json,
)

CATALOG = REPOSITORY / "deploy/nemotron_native/programs.json"
BATCHING_MODES = ("fixed_cohort", "fixed_request", "inflight")
INFLIGHT_PROFILES = ("retained", "playback-safe")


def playback_profiles():
    path = REPOSITORY / "deploy/nemotron_native/profiles/playback-safe.json"
    value = json.loads(path.read_text())
    policy = {"playback_reserve_guard": True, "codec_deadline_priority": True}
    expected = {
        str(width): {
            "physical_width": width,
            "policy": {**policy, **({"retire_batch": True} if width == 1024 else {})},
            "environment": {
                "NEMOTRON_INITIAL_AUDIO_HOLD_MS": hold,
                "NEMOTRON_MAX_ACTIVE_REQUESTS": cap,
            },
        }
        for width, hold, cap in ((1024, "40", None), (2048, "300", "1024"))
    }
    if value != {"schema": "nemotron.playback-profile.v1", "profiles": expected}:
        raise ValueError("Unsupported playback deployment profile")
    return expected


def profile_policy(width, batching_mode, profile):
    if profile not in INFLIGHT_PROFILES:
        raise ValueError("Unsupported in-flight profile")
    if profile == "retained":
        return {}
    if batching_mode != "inflight":
        raise ValueError("Playback deployment profile requires inflight batching")
    return playback_profiles().get(str(width), {}).get("policy", {})


def source_seal():
    root = REPOSITORY / "python/nemotron_native_backend"
    return {
        "package_source_pins": {
            str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*.py"))
        }
    }


def load_catalog(manifest):
    sys.path.insert(0, str(REPOSITORY / "python"))
    try:
        from nemotron_native_backend.execution.inflight_policy import InflightPolicy
    finally:
        sys.path.pop(0)

    catalog = json.loads(CATALOG.read_text())
    if (
        catalog.get("schema") != "nemotron.native-program-catalog.v1"
        or catalog.get("architecture") != manifest["architecture"]
        or set(catalog["programs"]) != {str(width) for width in SUPPORTED_WIDTHS}
    ):
        raise ValueError("Unsupported native program catalog")
    recipes = {recipe["output"]: recipe for recipe in manifest["recipes"].values()}
    for key, program in catalog["programs"].items():
        width = int(key)
        host = program["serving"]["host"]
        if "inflight_policy" in host or host.get("batching_mode", "fixed_cohort") != "fixed_cohort":
            raise ValueError("Program catalog must preserve the fixed-cohort serving defaults")
        InflightPolicy.parse(program.get("inflight_policy"))
        if (
            program["neural"]["physical_width"] != width
            or program["serving"]["physical_width"] != width
        ):
            raise ValueError("Inconsistent physical width")
        for role, path in program["libraries"].items():
            recipe = recipes[path]
            if recipe["width"] != width or recipe["role"] != role:
                raise ValueError("Program changes kernel binding: " + path)
    return catalog


def validate_build(build_root, manifest, catalog, widths):
    build = json.loads((build_root / "BUILD.json").read_text())
    if (
        build.get("schema") != "nemotron-native-kernel-build.v1"
        or build.get("status") != "BUILT_UNQUALIFIED"
        or build.get("source_manifest_sha256") != digest(SOURCE_ROOT / "recipes.json")
        or build.get("architecture") != manifest["architecture"]
        or not set(widths) <= set(build["widths"])
        or not build.get("nvcc_version")
    ):
        raise ValueError("Missing, failed, or stale native kernel build")
    for width in widths:
        for relative in catalog["programs"][str(width)]["libraries"].values():
            row = build["library_map"][relative]
            recipe = manifest["recipes"][row["recipe"]]
            path = confined(build_root, relative)
            if (
                row["status"] != "BUILT_UNQUALIFIED"
                or row["returncode"] != 0
                or row["path"] != relative
                or row["source_manifest_sha256"] != build["source_manifest_sha256"]
                or recipe["output"] != relative
                or row.get("bytes") != path.stat().st_size
                or digest(path) != row["sha256"]
            ):
                raise ValueError("Invalid compiled library: " + relative)
            compiled = manifest["recipes"][recipe.get("alias_of", row["recipe"])]
            argv = row.get("command")
            if (
                not isinstance(argv, list)
                or len(argv) != 4 + len(compiled["flags"]) + len(compiled["include_dirs"])
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)
                or argv[2] != "-o"
            ):
                raise ValueError("Invalid compiler command: " + relative)
            roots = []
            for actual, expected in ((argv[1], compiled["source"]), (argv[3], compiled["output"])):
                absolute = Path(actual)
                suffix = Path(expected)
                if (
                    not absolute.is_absolute()
                    or ".." in absolute.parts
                    or absolute.parts[-len(suffix.parts) :] != suffix.parts
                    or len(absolute.parts) <= len(suffix.parts)
                ):
                    raise ValueError("Compiler path differs from recipe: " + relative)
                roots.append(absolute.parents[len(suffix.parts) - 1])
            source_root, output_root = roots
            expected_argv = [
                argv[0],
                str(source_root / compiled["source"]),
                "-o",
                str(output_root / compiled["output"]),
                *compiled["flags"],
                *["-I" + str(source_root / entry) for entry in compiled["include_dirs"]],
            ]
            if argv != expected_argv:
                raise ValueError("Compiler command differs from recipe: " + relative)
            with path.open("rb") as stream:
                header = stream.read(64)
            if (
                len(header) != 64
                or header[:6] != b"\x7fELF\x02\x01"
                or struct.unpack_from("<HH", header, 16) != (3, 62)
                or struct.unpack_from("<H", header, 52)[0] != 64
                or struct.unpack_from("<H", header, 54)[0] != 56
                or struct.unpack_from("<H", header, 56)[0] == 0
                or struct.unpack_from("<Q", header, 32)[0]
                + 56 * struct.unpack_from("<H", header, 56)[0]
                > path.stat().st_size
            ):
                raise ValueError("Native library is not a complete x86-64 shared ELF: " + relative)
    return build


def state_artifacts(root):
    state = root / "state/bf16"
    state.mkdir(parents=True, exist_ok=True)
    scales = state / "scales.f32"
    mapping = state / "map.i32"
    scales.write_bytes(struct.pack("<f", 1.0) * (14 * 16384))
    mapping.write_bytes(struct.pack("<1536i", *range(1536)) * 14)
    recipe = state / "recipe.json"
    write_json(
        recipe,
        {
            "mode": 1,
            "reordering": "identity",
            "scale_usage": "unused-in-bf16-mode",
            "scales": {"sha256": digest(scales)},
            "map": {"sha256": digest(mapping)},
        },
    )
    return {"state_scales": scales, "state_map": mapping, "state_recipe": recipe}


def compiled_binding(neural):
    options = {
        "binding_module": "state_r9.binding",
        "batched_handoff": True,
        "pinned_pcm": True,
        "early_proof": True,
        "engine_binding": "integration_r9.binding",
        "engine_bindings": ["control_r9.chunk", "control_r9.prefix"],
        "transport_binding": "publication.wide_delivery",
    }
    if neural["kernels"]["prefill"].get("validation_kind") == "retained-device-dedup":
        options.pop("transport_binding")
    return {
        "id": neural["id"],
        "batch": neural["physical_width"],
        "physical_batch": neural["physical_width"],
        "neural_cohort_partitioning": False,
        "state_mode": neural["state"]["mode"],
        "state_format": neural["state"]["format"],
        "ctas": neural["execution"]["stage0_grid"],
        "prefill_workspace_sequences": neural["kernels"]["prefill"]["workspace_sequences"],
        "runtime_options": options,
        "r9_components": ["canonical-fp32-private-bf16", "prefix-cache", "pinned-pcm"],
    }


def assemble_program(
    output,
    build_root,
    build,
    manifest,
    catalog,
    width,
    batching_mode="fixed_cohort",
    inflight_profile="retained",
):
    overrides = profile_policy(width, batching_mode, inflight_profile)
    program = catalog["programs"][str(width)]
    neural = copy.deepcopy(program["neural"])
    serving = copy.deepcopy(program["serving"])
    if batching_mode not in BATCHING_MODES:
        raise ValueError("Unsupported request batching mode")
    if batching_mode != "fixed_cohort":
        if "stage0_inflight" not in program["libraries"]:
            raise ValueError("Request batching requires a source-bound masked producer")
        serving["host"]["batching_mode"] = batching_mode
        serving["host"]["inflight_policy"] = copy.deepcopy(program.get("inflight_policy", {}))
        serving["host"]["inflight_policy"].update(overrides)
    directory = output / "programs" / f"b{width}"
    root = directory / "artifacts"
    root.mkdir(parents=True)
    receipt_path = root / f"source-build-b{width}.json"
    artifacts = {}
    libraries = {}
    source_files = {p for p in manifest["files"] if p.startswith("third_party/")}
    for role, relative in program["libraries"].items():
        row = build["library_map"][relative]
        recipe = manifest["recipes"][row["recipe"]]
        source_files.update(recipe["dependencies"])
        source_files.add(recipe["source"])
        library = confined(output, relative)
        if not library.is_relative_to(root):
            raise ValueError("Library belongs to another physical program")
        library.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(confined(build_root, relative), library)
        receipt_name = "build-receipt.json" if role == "codec" else role + "-build-receipt.json"
        compiler_receipt = library.parent / receipt_name
        write_json(
            compiler_receipt,
            {
                "schema": "nemotron.native-source-library.v1",
                "status": "COMPILED",
                "compile_returncode": row["returncode"],
                "physical_width": width,
                "region": role,
                "library_sha256": digest(library),
                "command": row["command"],
                "compiler": build["nvcc_version"],
                "source_build_receipt": os.path.relpath(receipt_path, compiler_receipt.parent),
            },
        )
        artifacts[role] = library
        libraries[role] = {
            "path": str(library.relative_to(root)),
            "sha256": digest(library),
            "receipt": str(compiler_receipt.relative_to(root)),
            "receipt_sha256": digest(compiler_receipt),
        }
    sources = []
    for relative in sorted(source_files):
        target = confined(root, "sources/" + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(confined(SOURCE_ROOT, relative), target)
        sources.append({"path": str(target.relative_to(root)), "sha256": digest(target)})
    receipt = {
        "schema": "nemotron.native-source-build.v1",
        "status": "COMPILED",
        "physical_width": width,
        "source_manifest_sha256": build["source_manifest_sha256"],
        "sources": sources,
        "libraries": libraries,
    }
    if serving["family"] == "narrow":
        receipt["expected_residency"] = {
            "stage0": {"batch": width, "grid": neural["execution"]["full_stage0_grid"]},
            "codec_blocks": neural["execution"]["codec_grid"],
        }
    if serving["family"] == "wide":
        artifacts.update(state_artifacts(root))
        receipt["compiled_binding"] = compiled_binding(neural)
    write_json(receipt_path, receipt)
    artifacts["source_build_receipt"] = receipt_path
    neural["artifacts"] = {
        role: {"path": str(path.relative_to(root)), "sha256": digest(path)}
        for role, path in artifacts.items()
    }
    if neural.get("active_lanes", {}).get("mode") == "fixed-subset":
        for role in ("stage0", "transfer"):
            neural["active_lanes"][role] = copy.deepcopy(neural["artifacts"]["contraction_" + role])
    write_json(directory / "neural.json", neural)
    neural_hash = digest(directory / "neural.json")
    research = {
        "schema": "nemotron.chunk-schedule.program.r30.v1",
        "id": f"native-source-b{width}-startup2345",
        "physical_width": width,
        "family": serving["family"],
        "startup": catalog["startup"],
        "steady": catalog["steady"],
        "speech_delay": 5,
        "horizon": 511,
        "parent_generation_cap": 511,
        "parent_manifest_sha256": neural_hash,
        "parent_execution": neural["execution"],
        "parent_state": neural["state"],
        "qualification_only": True,
        "serving_enabled": False,
        "continuation_proof": True,
        "neural_dso_changes": False,
    }
    write_json(directory / "research.json", research)
    serving["neural_manifest_sha256"] = neural_hash
    serving["research_chunk_schedule"] = {
        "program_sha256": digest(directory / "research.json"),
        "startup": catalog["startup"],
        "steady": catalog["steady"],
        "horizon": 511,
        "qualification_only": True,
    }
    write_json(directory / "serving.json", serving)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=parse_widths, default=list(SUPPORTED_WIDTHS))
    parser.add_argument("--batching-mode", choices=BATCHING_MODES, default="fixed_cohort")
    parser.add_argument("--inflight-profile", choices=INFLIGHT_PROFILES, default="retained")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    catalog = load_catalog(manifest)
    widths = sorted(set(args.widths) | {1})
    if args.inflight_profile == "playback-safe" and not {1024, 2048}.intersection(widths):
        raise ValueError("Playback deployment profile requires width 1024 or 2048")
    for width in widths:
        profile_policy(width, args.batching_mode, args.inflight_profile)
    output = args.output.resolve()
    if output.is_relative_to(SOURCE_ROOT) or REPOSITORY.is_relative_to(output):
        raise ValueError("Package output overlaps source files")
    plan = {
        "schema": "nemotron.native-package-plan.v1",
        "status": "PLANNED",
        "widths": widths,
        "batching_mode": args.batching_mode,
        "inflight_profile": args.inflight_profile,
        "source_manifest_sha256": digest(SOURCE_ROOT / "recipes.json"),
        "program_catalog_sha256": digest(CATALOG),
    }
    if args.inflight_profile == "playback-safe":
        plan["deployment_profiles"] = {
            key: value for key, value in playback_profiles().items() if int(key) in widths
        }
    if args.plan_only:
        write_json(output / "PLAN.json", plan)
        print(json.dumps(plan))
        return 0
    if args.kernels is None:
        parser.error("--kernels is required unless --plan-only")
    build_root = args.kernels.resolve(strict=True)
    if (
        output == build_root
        or output.is_relative_to(build_root)
        or build_root.is_relative_to(output)
    ):
        raise ValueError("Package output overlaps compiler outputs")
    build = validate_build(build_root, manifest, catalog, widths)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Package destination must be new or empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-package-", dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        for width in widths:
            assemble_program(
                staged,
                build_root,
                build,
                manifest,
                catalog,
                width,
                args.batching_mode,
                args.inflight_profile,
            )
        write_json(staged / "source-seal.json", source_seal())
        write_json(staged / "PACKAGE.json", dict(plan, status="BUILT_UNQUALIFIED"))
        if output.exists():
            output.rmdir()
        staged.replace(output)
    print(json.dumps({"status": "BUILT_UNQUALIFIED", "package": str(output), "widths": widths}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
