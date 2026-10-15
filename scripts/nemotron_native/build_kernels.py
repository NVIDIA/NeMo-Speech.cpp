import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "deploy/nemotron_native/kernels"
SCHEMA = "nemotron-native-kernel-recipes.v2"
SUPPORTED_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {value!r}")
    return target


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_manifest(root=SOURCE_ROOT):
    path = root / "recipes.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("architecture") != "sm_90a":
        raise ValueError("Unsupported kernel recipe schema or architecture")
    files = manifest["files"]
    if not files:
        raise ValueError("Empty source manifest")
    for relative, expected in files.items():
        path = confined(root, relative)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid source digest: {relative}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed kernel source: {relative}")
    recipes = manifest["recipes"]
    if {r["width"] for r in recipes.values()} != set(SUPPORTED_WIDTHS):
        raise ValueError("Incomplete physical-width catalog")
    destinations = set()
    for name, recipe in recipes.items():
        for relative in [recipe["source"], *recipe["dependencies"]]:
            if relative not in files:
                raise ValueError(f"Unpinned source for {name}: {relative}")
        for relative in recipe["include_dirs"]:
            if not confined(root, relative).is_dir():
                raise ValueError(f"Missing include directory: {relative}")
        destination = recipe["output"]
        confined(Path("/output"), destination)
        if destination in destinations:
            raise ValueError(f"Duplicate library destination: {destination}")
        destinations.add(destination)
        flags = recipe["flags"]
        if not flags or any(not isinstance(x, str) for x in flags):
            raise ValueError(f"Invalid compiler arguments: {name}")
        if any(x.startswith(("sha256:", "-I/")) or x in ("-o", "--output-file") for x in flags):
            raise ValueError(f"Nonportable compiler arguments: {name}")
        alias = recipe.get("alias_of")
        if alias is not None:
            if alias not in recipes or recipes[alias].get("alias_of"):
                raise ValueError(f"Invalid alias: {name}")
            target = recipes[alias]
            if any(recipe[key] != target[key] for key in ("source", "flags", "include_dirs")):
                raise ValueError(f"Alias changes compiled implementation: {name}")
        for relative in recipe["dependencies"]:
            path = confined(root, relative)
            for include in re.findall(
                r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.MULTILINE
            ):
                candidates = [
                    path.parent / include,
                    *[root / d / include for d in recipe["include_dirs"]],
                ]
                matched = next((p for p in candidates if p.is_file()), None)
                if matched is None or not matched.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unresolved local include in {relative}: {include}")
                if str(matched.resolve().relative_to(root.resolve())) not in files:
                    raise ValueError(f"Unpinned local include in {relative}: {include}")
    return manifest


def select_recipes(manifest, widths):
    selected = {k: v for k, v in manifest["recipes"].items() if v["width"] in widths}
    required = dict(selected)
    for recipe in selected.values():
        alias = recipe.get("alias_of")
        if alias:
            required[alias] = manifest["recipes"][alias]
    return selected, required


def command(recipe, output, nvcc, root=SOURCE_ROOT):
    source = confined(root, recipe["source"])
    destination = confined(output, recipe["output"])
    return [
        nvcc,
        str(source),
        "-o",
        str(destination),
        *recipe["flags"],
        *["-I" + str(confined(root, p)) for p in recipe["include_dirs"]],
    ]


def parse_widths(value):
    try:
        widths = sorted({int(x.removeprefix("b")) for x in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("Widths must be comma-separated integers") from error
    if not widths or any(x not in SUPPORTED_WIDTHS for x in widths):
        raise argparse.ArgumentTypeError("Unsupported physical width")
    return widths


def compile_one(name, recipe, output, nvcc, timeout, source_digest):
    destination = confined(output, recipe["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = confined(output, "logs/" + name.replace("/", "-") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = command(recipe, output, nvcc)
    start = time.monotonic()
    result = {
        "recipe": name,
        "path": recipe["output"],
        "source_manifest_sha256": source_digest,
        "command": argv,
        "log": str(log.relative_to(output)),
        "status": "FAILED",
    }
    try:
        with log.open("w") as handle:
            process = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        result["returncode"] = returncode
        if returncode != 0:
            return result
        if not destination.is_file() or destination.read_bytes()[:4] != b"\x7fELF":
            result["error"] = "Compiler did not create an ELF library"
            return result
        result.update(
            status="BUILT_UNQUALIFIED", sha256=digest(destination), bytes=destination.stat().st_size
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    finally:
        result["seconds"] = time.monotonic() - start
        write_json(output / "receipts" / (name.replace("/", "-") + ".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--widths", type=parse_widths, default=list(SUPPORTED_WIDTHS))
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    args.widths = sorted({1, *args.widths})
    if not 1 <= args.jobs <= 8 or args.timeout <= 0:
        parser.error("jobs must be 1..8 and timeout must be positive")
    if args.output is None and not args.validate:
        parser.error("--output is required")
    manifest = load_manifest()
    selected, required = select_recipes(manifest, args.widths)
    source_digest = digest(SOURCE_ROOT / "recipes.json")
    if args.validate:
        print(
            json.dumps(
                {
                    "status": "PASS_SOURCE_CLOSURE",
                    "recipes": len(selected),
                    "source_files": len(manifest["files"]),
                    "source_manifest_sha256": source_digest,
                }
            )
        )
        return 0
    output = args.output.resolve()
    if output.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(output):
        raise ValueError("Build output cannot overlap kernel sources")
    if (output / "BUILD.json").exists():
        raise ValueError("Build result already exists; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "nemotron-native-kernel-build.v1",
        "status": "PLANNED",
        "architecture": manifest["architecture"],
        "widths": args.widths,
        "source_manifest_sha256": source_digest,
        "builder_sha256": digest(Path(__file__)),
        "recipes": {
            k: {**v, "command": command(v, output, args.nvcc)} for k, v in required.items()
        },
        "library_map": {
            v["output"]: {"recipe": k, "path": v["output"]} for k, v in selected.items()
        },
    }
    write_json(output / "PLAN.json", plan)
    if args.plan_only:
        print(
            json.dumps(
                {"status": "PLANNED", "recipes": len(selected), "plan": str(output / "PLAN.json")}
            )
        )
        return 0
    version = subprocess.run(
        [args.nvcc, "--version"], capture_output=True, text=True, timeout=30, check=True
    )
    plan["nvcc_version"] = version.stdout.strip()
    pending = {k: v for k, v in required.items() if not v.get("alias_of")}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(compile_one, k, v, output, args.nvcc, args.timeout, source_digest): k
            for k, v in pending.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(
                json.dumps(
                    {
                        "recipe": name,
                        "status": results[name]["status"],
                        "seconds": results[name]["seconds"],
                    }
                ),
                flush=True,
            )
    for name, recipe in selected.items():
        if not recipe.get("alias_of"):
            continue
        original = results[recipe["alias_of"]]
        result = {
            **original,
            "recipe": name,
            "path": recipe["output"],
            "alias_of": recipe["alias_of"],
            "seconds": 0.0,
        }
        if original["status"] == "BUILT_UNQUALIFIED":
            destination = confined(output, recipe["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(confined(output, original["path"]), destination)
        results[name] = result
        write_json(output / "receipts" / (name.replace("/", "-") + ".json"), result)
    library_map = {selected[k]["output"]: results[k] for k in selected}
    status = (
        "BUILT_UNQUALIFIED"
        if all(v["status"] == "BUILT_UNQUALIFIED" for v in library_map.values())
        else "FAILED"
    )
    write_json(
        output / "BUILD.json",
        {**plan, "status": status, "library_map": library_map, "results": results},
    )
    return 0 if status == "BUILT_UNQUALIFIED" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
