#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import http.server
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse

PAYLOAD = b"NeMo-Speech.cpp model-store fixture\n"
CODEC_PAYLOAD = b"NeMo-Speech.cpp codec fixture\n"
TTS_PAYLOAD = b"NeMo-Speech.cpp TTS fixture\n"
TOKENIZER_PAYLOAD = b"tokenizer configuration\n"
TTS_REVISION = "1" * 40
TOKENIZER_REVISION = "3" * 40
CODEC_REVISION = "2" * 40
ASR_ARTIFACT_REVISION = "4" * 40
STALE_ASR_ARTIFACT_REVISION = "5" * 40


class ArtifactHandler(http.server.BaseHTTPRequestHandler):
    requests = 0
    request_counts: dict[str, int] = {}
    request_paths: list[str] = []
    payloads: dict[str, bytes] = {}

    def do_GET(self) -> None:
        type(self).requests += 1
        request_path = urllib.parse.urlsplit(self.path).path
        type(self).request_paths.append(request_path)
        filename = pathlib.PurePosixPath(request_path).name
        type(self).request_counts[filename] = type(self).request_counts.get(filename, 0) + 1
        payload = type(self).payloads.get(filename)
        if payload is None:
            self.send_error(404)
            return
        start = 0
        end = len(payload) - 1
        range_header = self.headers.get("Range")
        if range_header:
            assert range_header.startswith("bytes=")
            bounds = range_header.removeprefix("bytes=").split("-", 1)
            start = int(bounds[0])
            if bounds[1]:
                end = min(end, int(bounds[1]))
        body = payload[start : end + 1]
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Length", str(len(body)))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def run(binary: str, environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, *arguments],
        env=environment,
        check=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tokenizer_archive() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as bundle:
        tokenizer = tarfile.TarInfo("tokenizer.txt")
        tokenizer.size = len(TOKENIZER_PAYLOAD)
        tokenizer.pax_headers = {"mtime": "0.0"}
        bundle.addfile(tokenizer, io.BytesIO(TOKENIZER_PAYLOAD))
        weights = tarfile.TarInfo("model_weights.ckpt")
        weights.size = 1
        bundle.addfile(weights, io.BytesIO(b"x"))
    return output.getvalue()


def file_artifact(role: str, filename: str, payload: bytes, sha256: str | None = None) -> dict:
    return {
        "role": role,
        "type": "file",
        "filename": filename,
        "size": len(payload),
        "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
    }


def write_index(
    path: pathlib.Path,
    asr_sha256: str,
    tokenizer_tar: bytes,
    asr_payload: bytes = PAYLOAD,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {
                    "asr": "acme/tiny-asr",
                    "tts": "acme/tiny-tts",
                    "tokenizer": "acme/tiny-tts",
                    "codec": "acme/tiny-codec",
                },
                "models": [
                    {
                        "repo": "acme/tiny-asr",
                        "aliases": ["tiny-asr"],
                        "revision": "0" * 40,
                        "license": "Test only",
                        "license_url": "https://example.invalid/license",
                        "artifacts": [file_artifact("asr", "tiny.gguf", asr_payload, asr_sha256)],
                    },
                    {
                        "repo": "acme/tiny-tts",
                        "aliases": ["tiny-tts"],
                        "revision": TTS_REVISION,
                        "license": "Test only",
                        "license_url": "https://example.invalid/license",
                        "companions": ["acme/tiny-codec"],
                        "artifacts": [
                            file_artifact("tts", "tiny-tts.gguf", TTS_PAYLOAD),
                            {
                                "role": "tokenizer",
                                "type": "tar-prefix",
                                "filename": "tiny-tts.nemo",
                                "revision": TOKENIZER_REVISION,
                                "directory": "tokenizer",
                                "size": len(tokenizer_tar),
                                "range_end": len(tokenizer_tar) - 1,
                                "sha256": hashlib.sha256(tokenizer_tar).hexdigest(),
                                "stop_before": "model_weights.ckpt",
                                "members": [
                                    {
                                        "name": "tokenizer.txt",
                                        "size": len(TOKENIZER_PAYLOAD),
                                        "sha256": hashlib.sha256(TOKENIZER_PAYLOAD).hexdigest(),
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "repo": "acme/tiny-codec",
                        "aliases": ["tiny-codec"],
                        "revision": CODEC_REVISION,
                        "license": "Test only",
                        "license_url": "https://example.invalid/license",
                        "artifacts": [file_artifact("codec", "tiny-codec.gguf", CODEC_PAYLOAD)],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def expect_json_error(result: subprocess.CompletedProcess, code: int) -> dict:
    assert result.returncode == code, result.stdout + result.stderr
    assert result.stdout == ""
    value = json.loads(result.stderr)
    assert value["error"]["exit_code"] == code
    return value["error"]


def main() -> None:
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="nemo-speech-model-store-") as temporary:
        root = pathlib.Path(temporary)
        index = root / "indéx.json"
        cache = root / "caché"
        tokenizer_tar = tokenizer_archive()
        write_index(index, hashlib.sha256(PAYLOAD).hexdigest(), tokenizer_tar)
        ArtifactHandler.requests = 0
        ArtifactHandler.request_counts = {}
        ArtifactHandler.request_paths = []
        ArtifactHandler.payloads = {
            "tiny.gguf": PAYLOAD,
            "tiny-tts.gguf": TTS_PAYLOAD,
            "tiny-tts.nemo": tokenizer_tar,
            "tiny-codec.gguf": CODEC_PAYLOAD,
        }
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ArtifactHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        environment = os.environ.copy()
        environment.update(
            {
                "NEMO_SPEECH_MODEL_INDEX": str(index),
                "NEMO_SPEECH_MODEL_DIR": str(cache),
                "NEMO_SPEECH_HF_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            }
        )
        try:
            plain_listing = run(binary, environment, "model", "list")
            assert plain_listing.returncode == 0, plain_listing.stderr
            assert "ASR — transcribe, bench, serve" in plain_listing.stdout
            assert "Diarization — diarize, transcribe --diarize, serve" in plain_listing.stdout
            assert "TTS — synthesize, serve" in plain_listing.stdout
            assert "* tiny-asr" in plain_listing.stdout
            assert "repo: acme/tiny-asr" in plain_listing.stdout

            listed = run(binary, environment, "--json", "model", "list")
            assert listed.returncode == 0, listed.stderr
            listing = json.loads(listed.stdout)
            assert listing["defaults"]["asr"] == "acme/tiny-asr"
            assert listing["models"][0]["aliases"] == ["tiny-asr"]
            assert listing["models"][0]["commands"] == ["transcribe", "bench", "serve"]
            assert listing["models"][0]["default_for"] == ["asr"]
            assert listing["models"][1]["commands"] == ["synthesize", "serve"]
            assert listing["models"][1]["default_for"] == ["tokenizer", "tts"]

            pulled = run(binary, environment, "--json", "pull", "tiny-asr")
            assert pulled.returncode == 0, pulled.stderr
            artifact = json.loads(pulled.stdout)["artifacts"][0]
            destination = pathlib.Path(artifact["path"])
            assert artifact["repo"] == "acme/tiny-asr"
            assert artifact["cached"] is False
            assert destination.read_bytes() == PAYLOAD
            marker = pathlib.Path(f"{destination}.verified")
            marker_lines = marker.read_text(encoding="utf-8").splitlines()
            assert marker_lines[0] == f"sha256={hashlib.sha256(PAYLOAD).hexdigest()}"
            assert marker_lines[1] == f"size={len(PAYLOAD)}"
            assert marker_lines[2].startswith("mtime=")
            requests_after_pull = ArtifactHandler.requests

            cached = run(binary, environment, "--json", "model", "pull", "acme/tiny-asr")
            assert cached.returncode == 0, cached.stderr
            assert json.loads(cached.stdout)["artifacts"][0]["cached"] is True
            assert ArtifactHandler.requests == requests_after_pull

            marker.write_text("invalid\n", encoding="utf-8")
            refreshed = run(binary, environment, "--json", "pull", "tiny-asr")
            assert refreshed.returncode == 0, refreshed.stderr
            assert json.loads(refreshed.stdout)["artifacts"][0]["cached"] is True
            assert ArtifactHandler.requests == requests_after_pull
            assert marker.read_text(encoding="utf-8").splitlines() == marker_lines

            tts = run(binary, environment, "--json", "pull", "tiny-tts")
            assert tts.returncode == 0, tts.stderr
            tts_artifacts = json.loads(tts.stdout)["artifacts"]
            assert [(item["repo"], item["role"]) for item in tts_artifacts] == [
                ("acme/tiny-tts", "tts"),
                ("acme/tiny-tts", "tokenizer"),
                ("acme/tiny-codec", "codec"),
            ]
            tokenizer = pathlib.Path(tts_artifacts[1]["path"])
            assert (tokenizer / "tokenizer.txt").read_bytes() == TOKENIZER_PAYLOAD
            assert not (tokenizer / "model_weights.ckpt").exists()
            assert ArtifactHandler.request_counts["tiny-tts.gguf"] == 1
            assert ArtifactHandler.request_counts["tiny-tts.nemo"] == 1
            assert ArtifactHandler.request_counts["tiny-codec.gguf"] == 1
            assert (
                f"/acme/tiny-tts/resolve/{TTS_REVISION}/tiny-tts.gguf"
                in ArtifactHandler.request_paths
            )
            assert (
                f"/acme/tiny-tts/resolve/{TOKENIZER_REVISION}/tiny-tts.nemo"
                in ArtifactHandler.request_paths
            )
            assert (
                f"/acme/tiny-codec/resolve/{CODEC_REVISION}/tiny-codec.gguf"
                in ArtifactHandler.request_paths
            )

            cached_tts = run(binary, environment, "--json", "model", "pull", "acme/tiny-tts")
            assert cached_tts.returncode == 0, cached_tts.stderr
            assert all(item["cached"] for item in json.loads(cached_tts.stdout)["artifacts"])
            assert ArtifactHandler.request_counts["tiny-tts.gguf"] == 1
            assert ArtifactHandler.request_counts["tiny-tts.nemo"] == 1
            assert ArtifactHandler.request_counts["tiny-codec.gguf"] == 1

            (tokenizer / "tokenizer.txt").write_bytes(b"x" * len(TOKENIZER_PAYLOAD))
            refreshed_tts = run(binary, environment, "--json", "pull", "tiny-tts")
            assert refreshed_tts.returncode == 0, refreshed_tts.stderr
            refreshed_artifacts = json.loads(refreshed_tts.stdout)["artifacts"]
            assert [item["cached"] for item in refreshed_artifacts] == [True, False, True]
            assert (tokenizer / "tokenizer.txt").read_bytes() == TOKENIZER_PAYLOAD
            assert ArtifactHandler.request_counts["tiny-tts.gguf"] == 1
            assert ArtifactHandler.request_counts["tiny-tts.nemo"] == 2
            assert ArtifactHandler.request_counts["tiny-codec.gguf"] == 1

            previous_mtime = destination.stat().st_mtime_ns
            destination.write_bytes(b"x" * len(PAYLOAD))
            os.utime(destination, ns=(previous_mtime + 2_000_000_000,) * 2)
            repaired = run(binary, environment, "--json", "pull", "tiny-asr")
            assert repaired.returncode == 0, repaired.stderr
            assert json.loads(repaired.stdout)["artifacts"][0]["cached"] is False
            assert destination.read_bytes() == PAYLOAD
            assert marker.read_text(encoding="utf-8").splitlines() != marker_lines

            revision_index = json.loads(index.read_text(encoding="utf-8"))
            assert revision_index["models"][0]["revision"] == "0" * 40
            revision_index["models"][0]["artifacts"][0]["revision"] = ASR_ARTIFACT_REVISION
            index.write_text(json.dumps(revision_index), encoding="utf-8")
            revision_cache = root / "revision-cache"
            revision_partial = (
                revision_cache / "acme" / "tiny-asr" / ("0" * 40) / "tiny.gguf.partial"
            )
            revision_partial.parent.mkdir(parents=True)
            revision_partial.write_bytes(b"x" * 5)
            pathlib.Path(f"{revision_partial}.revision").write_text(
                f"{STALE_ASR_ARTIFACT_REVISION}\n", encoding="utf-8"
            )
            revision_environment = {
                **environment,
                "NEMO_SPEECH_MODEL_DIR": str(revision_cache),
            }
            revision_pull = run(binary, revision_environment, "--json", "pull", "tiny-asr")
            assert revision_pull.returncode == 0, revision_pull.stderr
            revision_artifact = json.loads(revision_pull.stdout)["artifacts"][0]
            assert pathlib.Path(revision_artifact["path"]).read_bytes() == PAYLOAD
            assert (
                f"/acme/tiny-asr/resolve/{ASR_ARTIFACT_REVISION}/tiny.gguf"
                in ArtifactHandler.request_paths
            )
            assert not revision_partial.exists()
            assert not pathlib.Path(f"{revision_partial}.revision").exists()
            write_index(index, hashlib.sha256(PAYLOAD).hexdigest(), tokenizer_tar)

            unstable_payload = b"x" * (16 * 1024 * 1024)
            write_index(
                index,
                hashlib.sha256(unstable_payload).hexdigest(),
                tokenizer_tar,
                unstable_payload,
            )
            unstable_cache = root / "unstable-cache"
            unstable_partial = (
                unstable_cache / "acme" / "tiny-asr" / ("0" * 40) / "tiny.gguf.partial"
            )
            unstable_partial.parent.mkdir(parents=True)
            unstable_partial.write_bytes(unstable_payload)
            pathlib.Path(f"{unstable_partial}.revision").write_text(
                f"{'0' * 40}\n", encoding="utf-8"
            )
            unstable_environment = {
                **environment,
                "NEMO_SPEECH_MODEL_DIR": str(unstable_cache),
            }
            requests_before_unstable_pull = ArtifactHandler.requests

            def change_partial_mtime() -> None:
                deadline = time.monotonic() + 0.2
                while time.monotonic() < deadline:
                    try:
                        os.utime(unstable_partial, None)
                    except FileNotFoundError:
                        return
                    time.sleep(0.001)

            mtime_thread = threading.Thread(target=change_partial_mtime)
            mtime_thread.start()
            unstable_pull = run(binary, unstable_environment, "--json", "pull", "tiny-asr")
            mtime_thread.join(timeout=5)
            assert unstable_pull.returncode == 0, unstable_pull.stderr
            unstable_artifact = json.loads(unstable_pull.stdout)["artifacts"][0]
            assert pathlib.Path(unstable_artifact["path"]).read_bytes() == unstable_payload
            assert ArtifactHandler.requests == requests_before_unstable_pull
            write_index(index, hashlib.sha256(PAYLOAD).hexdigest(), tokenizer_tar)

            unknown = expect_json_error(
                run(binary, environment, "--json", "pull", "unknown/repository"), 3
            )
            assert unknown["type"] == "missing_model"

            write_index(index, "0" * 64, tokenizer_tar)
            bad_cache = root / "bad-cache"
            bad_environment = {**environment, "NEMO_SPEECH_MODEL_DIR": str(bad_cache)}
            invalid = expect_json_error(
                run(binary, bad_environment, "--json", "pull", "tiny-asr"), 1
            )
            assert "SHA-256 verification" in invalid["message"]
            assert f"revision {'0' * 40}" in invalid["message"]
            assert f"expected size {len(PAYLOAD)}" in invalid["message"]
            assert f"actual size {len(PAYLOAD)}" in invalid["message"]
            assert f"expected SHA-256 {'0' * 64}" in invalid["message"]
            assert f"actual SHA-256 {hashlib.sha256(PAYLOAD).hexdigest()}" in invalid["message"]
            assert not list(bad_cache.rglob("tiny.gguf"))
            assert not list(bad_cache.rglob("tiny.gguf.partial"))
            assert not list(bad_cache.rglob("tiny.gguf.partial.revision"))

            write_index(index, hashlib.sha256(PAYLOAD).hexdigest(), tokenizer_tar)
            oversized_payload = PAYLOAD + b"x"
            ArtifactHandler.payloads["tiny.gguf"] = oversized_payload
            oversized_cache = root / "oversized-cache"
            oversized_environment = {
                **environment,
                "NEMO_SPEECH_MODEL_DIR": str(oversized_cache),
            }
            oversized = expect_json_error(
                run(binary, oversized_environment, "--json", "pull", "tiny-asr"), 1
            )
            assert f"expected size {len(PAYLOAD)}" in oversized["message"]
            assert f"actual size {len(oversized_payload)}" in oversized["message"]
            assert f"expected SHA-256 {hashlib.sha256(PAYLOAD).hexdigest()}" in oversized["message"]
            assert (
                f"actual SHA-256 {hashlib.sha256(oversized_payload).hexdigest()}"
                in oversized["message"]
            )
            assert not list(oversized_cache.rglob("tiny.gguf"))
            assert not list(oversized_cache.rglob("tiny.gguf.partial"))
            assert not list(oversized_cache.rglob("tiny.gguf.partial.revision"))
            ArtifactHandler.payloads["tiny.gguf"] = PAYLOAD

            invalid_revision_index = json.loads(index.read_text(encoding="utf-8"))
            invalid_revision_index["models"][1]["artifacts"][1]["revision"] = "invalid"
            index.write_text(json.dumps(invalid_revision_index), encoding="utf-8")
            invalid_revision = expect_json_error(
                run(binary, environment, "--json", "pull", "tiny-tts"), 1
            )
            assert "revision must be a full commit SHA" in invalid_revision["message"]
            assert "acme/tiny-tts/tokenizer" in invalid_revision["message"]
            write_index(index, hashlib.sha256(PAYLOAD).hexdigest(), tokenizer_tar)

            missing_environment = {
                **environment,
                "PATH": str(root / "empty-path"),
                "NEMO_SPEECH_MODEL_DIR": str(root / "missing-curl-cache"),
            }
            missing = expect_json_error(
                run(binary, missing_environment, "--json", "pull", "tiny-asr"), 1
            )
            assert "curl executable" in missing["message"]
            assert "Local model paths" in missing["message"]

            unsafe_environment = {
                **environment,
                "NEMO_SPEECH_HF_BASE_URL": "http://localhost.example.invalid",
                "NEMO_SPEECH_MODEL_DIR": str(root / "unsafe-url-cache"),
            }
            unsafe = expect_json_error(
                run(binary, unsafe_environment, "--json", "pull", "tiny-asr"), 1
            )
            assert "must use HTTPS" in unsafe["message"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
