import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA = "nemotron-native-runtime-dependencies.v1"
MODULES = ("tensorrt", "tensorrt_llm", "torch_tensorrt")
REQUIRED = ("msgpack", "numpy", "safetensors", "tokenizers", "torch")
BACKEND_MODULES = (
    "nemotron_native_backend.worker",
    "nemotron_native_backend.execution.inflight_engine",
    "nemotron_native_backend.execution.native_stage1",
)
LIBRARY_ROOTS = (Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt"))
REMNANTS = (Path("/opt/tensorrt"), Path("/usr/src/tensorrt"))
SYSTEM_LIBRARIES = Path("/usr/lib/x86_64-linux-gnu")


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower().split(":", 1)[0]


def is_tensorrt_package(name):
    value = canonical_name(name)
    return (
        value == "polygraphy"
        or value.startswith(("tensorrt", "torch-tensorrt", "nvidia-tensorrt"))
        or value.startswith(("libnvinfer", "libnvonnxparsers", "python3-libnvinfer"))
    )


def is_tensorrt_library(name):
    return (
        Path(name)
        .name.lower()
        .startswith(
            (
                "libnvinfer",
                "libnvonnxparser",
                "libtensorrt",
                "libtorchtrt",
                "do_not_link_against_nvinfer",
            )
        )
    )


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def debian_packages():
    output = run("dpkg-query", "-W", "-f=${binary:Package}\t${db:Status-Status}\n")
    return sorted(
        name
        for line in output.splitlines()
        for name, status in [line.split("\t", 1)]
        if status == "installed" and is_tensorrt_package(name)
    )


def python_packages():
    return sorted(
        {d.metadata["Name"] for d in importlib.metadata.distributions() if d.metadata["Name"]}
    )


def validate_purge_plan(output, requested):
    selected = {canonical_name(name) for name in requested}
    proposed = {
        canonical_name(line.split()[1])
        for line in output.splitlines()
        if line.startswith(("Remv ", "Purg "))
    }
    if proposed != selected:
        raise RuntimeError(f"TensorRT purge changes unexpected packages: {proposed ^ selected}")


def remove_builder_resource_links(root):
    removed = []
    for path in sorted(root.glob("do_not_link_against_nvinfer_builder_resource_*")):
        if not path.is_symlink() or path.exists():
            raise RuntimeError(f"Expected an orphaned TensorRT builder resource link: {path}")
        if not path.readlink().name.startswith("libnvinfer_builder_resource_"):
            raise RuntimeError(f"Unexpected TensorRT resource link target: {path}")
        path.unlink()
        removed.append(str(path))
    return removed


def remove_tensorrt():
    python = [name for name in python_packages() if is_tensorrt_package(name)]
    debian = debian_packages()
    if python:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", *python], check=True)
    if debian:
        simulated = run("apt-get", "--simulate", "--no-auto-remove", "purge", *debian)
        validate_purge_plan(simulated, debian)
        subprocess.run(["apt-get", "--yes", "--no-auto-remove", "purge", *debian], check=True)
    removed = remove_builder_resource_links(SYSTEM_LIBRARIES)
    for path in REMNANTS:
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(str(path))
        elif path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    subprocess.run(["ldconfig"], check=True)
    return {"python_packages": python, "debian_packages": debian, "remnants": removed}


def find_tensorrt_libraries(roots):
    found = set()
    for root in roots:
        for directory, subdirectories, files in os.walk(root, followlinks=False):
            subdirectories[:] = [name for name in subdirectories if name != "__pycache__"]
            for name in files:
                if is_tensorrt_library(name):
                    found.add(str(Path(directory) / name))
    return sorted(found)


def needed_libraries(path):
    output = run("readelf", "--dynamic", str(path))
    return re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", output)


def audit_elf(path):
    path = Path(path)
    needed = needed_libraries(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "needed": needed,
        "tensorrt_needed": [name for name in needed if is_tensorrt_library(name)],
    }


def import_runtime(include_backend, required=REQUIRED):
    modules = [*required, *(BACKEND_MODULES if include_backend else ())]
    imported = {}
    for name in modules:
        module = importlib.import_module(name)
        imported[name] = getattr(module, "__version__", None)
    torch = sys.modules["torch"]
    if torch.cuda.is_initialized():
        raise RuntimeError("The CPU dependency audit unexpectedly initialized CUDA")
    return imported


def check(*, include_backend=False, library_roots=LIBRARY_ROOTS, elf_roots=(), required=REQUIRED):
    imported = import_runtime(include_backend, required)
    python = [name for name in python_packages() if is_tensorrt_package(name)]
    debian = debian_packages()
    modules = [name for name in MODULES if importlib.util.find_spec(name) is not None]
    loaded = [
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in MODULES)
    ]
    libraries = find_tensorrt_libraries(library_roots)
    loader = [
        line.strip()
        for line in run("ldconfig", "-p").splitlines()
        if line.strip() and is_tensorrt_library(line.split()[0])
    ]
    bad_environment = {
        name: value
        for name in ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONPATH")
        if any(part in (value := os.environ.get(name, "")).lower() for part in MODULES)
    }
    elf = []
    roots = [Path(sys.modules["torch"].__file__).parent / "lib", *map(Path, elf_roots)]
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"ELF audit directory does not exist: {root}")
        for path in sorted(root.rglob("*.so*")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    if handle.read(4) != b"\x7fELF":
                        continue
                elf.append(audit_elf(path))
    findings = {
        "python_packages": python,
        "debian_packages": debian,
        "importable_modules": modules,
        "loaded_modules": loaded,
        "library_files": libraries,
        "loader_cache": loader,
        "search_paths": bad_environment,
        "elf_dependencies": [row for row in elf if row["tensorrt_needed"]],
    }
    return {
        "schema": SCHEMA,
        "status": "PASS" if not any(findings.values()) else "FAIL",
        "scope": "CPU imports and dependency absence; no GPU inference qualification",
        "python": sys.version,
        "imports": imported,
        "torch_cuda_version": sys.modules["torch"].version.cuda,
        "cuda_initialized": False,
        "findings": findings,
        "elf": elf,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-tensorrt", action="store_true")
    parser.add_argument("--include-backend", action="store_true")
    parser.add_argument("--elf-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = {"schema": SCHEMA, "status": "FAIL"}
    try:
        removed = remove_tensorrt() if args.remove_tensorrt else None
        receipt = check(
            include_backend=args.include_backend,
            elf_roots=args.elf_root,
            required=("numpy", "torch") if args.remove_tensorrt else REQUIRED,
        )
        if removed is not None:
            receipt["removed"] = removed
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    summary = {"status": receipt["status"], "receipt": str(args.output)}
    if receipt["status"] != "PASS":
        summary.update(error=receipt.get("error"), findings=receipt.get("findings"))
    print(json.dumps(summary))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
