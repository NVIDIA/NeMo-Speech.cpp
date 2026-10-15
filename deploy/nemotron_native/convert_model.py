import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parent / "model.lock.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root, rows, label):
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
        if "bytes" in row and path.stat().st_size != row["bytes"]:
            raise SystemExit(f"{label} size mismatch: {path}")
        if "sha256" in row and sha256(path) != row["sha256"]:
            raise SystemExit(f"{label} sha256 mismatch: {path}")


def download(lock, inputs):
    source = lock["inputs"]
    if not os.environ.get(source["token_environment_variable"]):
        raise SystemExit(
            f"set {source['token_environment_variable']} to download {source['repository']}"
        )
    argv = ["hf", "download", source["repository"], "--repo-type", "model"]
    argv += ["--revision", source["revision"], "--local-dir", str(inputs)]
    argv += [row["path"] for row in source["files"]]
    subprocess.run(argv, check=True)


def checkout(lock, converter):
    spec = lock["converter"]
    if not (converter / ".git").exists():
        subprocess.run(["git", "init", "--quiet", str(converter)], check=True)
        subprocess.run(
            ["git", "-C", str(converter), "remote", "add", "origin", spec["repository"]], check=True
        )
    subprocess.run(
        ["git", "-C", str(converter), "fetch", "--depth", "1", "origin", spec["commit"]], check=True
    )
    subprocess.run(
        ["git", "-C", str(converter), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(converter), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != spec["commit"]:
        raise SystemExit("converter commit mismatch")
    if sha256(converter / spec["script"]) != spec["script_sha256"]:
        raise SystemExit("converter script sha256 mismatch")


def convert(lock, inputs, converter, output, gpu):
    spec = lock["converter"]
    output.mkdir(parents=True, exist_ok=True)
    install = " && ".join(
        f"python3 -m pip install --no-cache-dir {shlex.quote(r)}"
        for r in spec["extra_pip_requirements"]
    )
    command = " ".join(shlex.quote(a) for a in lock["argv"])
    script = f"{install} && {command}" if install else command
    argv = ["docker", "run", "--rm", "--gpus", f"device={gpu}", "--network", "bridge"]
    argv += ["--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp"]
    argv += [
        "--env",
        "PYTHONPATH=/repo",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--workdir",
        "/repo",
    ]
    argv += ["--volume", f"{converter}:/repo:ro", "--volume", f"{inputs}:/ckpt:ro"]
    argv += ["--volume", f"{output}:/out", "--entrypoint", "bash", spec["image"], "-c", script]
    subprocess.run(argv, check=True)


def seal(lock, output, path):
    verify(output, lock["outputs"], "model output")
    files = {row["path"]: dict(sha256=row["sha256"], bytes=row["bytes"]) for row in lock["outputs"]}
    value = dict(model_identity=lock["model_identity"], files=files)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args(argv)
    lock = json.loads(LOCK.read_text())
    output = args.output.resolve()
    if not args.seal_only:
        if args.inputs is None:
            parser.error("--inputs is required unless --seal-only")
        inputs = args.inputs.resolve()
        if args.converter is None:
            parser.error("--converter is required unless --seal-only")
        if args.download:
            download(lock, inputs)
        verify(inputs, lock["inputs"]["files"], "input")
        converter = args.converter.resolve()
        checkout(lock, converter)
        convert(lock, inputs, converter, output, args.gpu)
    written = seal(lock, output, (args.seal or output / "model-seal.json").resolve())
    print(
        json.dumps(dict(status="SEALED", model_identity=lock["model_identity"], seal=str(written)))
    )


if __name__ == "__main__":
    sys.exit(main())
