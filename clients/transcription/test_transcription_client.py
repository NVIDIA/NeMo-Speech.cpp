#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of transcription_client.py against a local fake of the API (no model, no GPU). Each test checks what the
openai SDK put on the wire and what the client made of the reply.
  python3 -m unittest discover -s clients/transcription -p 'test_*.py' -v"""

import contextlib
import email.parser
import email.policy
import io
import json
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcription_client as client  # noqa: E402

WORDS = [
    {"word": "Hello", "start": 0.4, "end": 0.8, "speaker": "spk:0"},
    {"word": "world.", "start": 0.8, "end": 1.3, "speaker": "spk:1"},
]


def multipart_fields(headers, body):
    """{name: [values]} of a multipart/form-data body; files become (filename, bytes)."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + headers["Content-Type"].encode() + b"\r\n\r\n" + body
    )
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        fields.setdefault(name, []).append((filename, payload) if filename else payload.decode())
    return fields


class FakeApi(BaseHTTPRequestHandler):
    """Answers like the NeMo Speech server; records every request in `server.requests`."""

    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def error(self, status, message, param=None, code="invalid_request"):
        self.reply(
            status,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": code,
                }
            },
        )

    def do_GET(self):
        self.server.requests.append(("GET", self.path, dict(self.headers), None))
        if self.path == "/v1/models":
            self.reply(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "denoiser", "object": "model", "capability": "speech"},
                        {"id": "asr-model", "object": "model", "capability": "transcription"},
                    ],
                },
            )
        else:
            self.reply(404, "")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
            fields = multipart_fields(self.headers, body)
            self.server.requests.append(("POST", self.path, dict(self.headers), fields))
            value = lambda name: (fields.get(name) or [None])[0]  # noqa: E731
            if value("language") == "xx":
                return self.error(400, "unsupported language: xx", None)
            fmt = value("response_format") or "json"
            if fmt in ("text", "srt", "vtt"):
                text = (
                    "Hello world."
                    if fmt == "text"
                    else "1\n00:00:00,400 --> 00:00:01,300\nHello world.\n\n"
                )
                return self.reply(
                    200, text, "text/plain; charset=utf-8" if fmt != "vtt" else "text/vtt"
                )
            out = {"text": "Hello world."}
            if fmt == "verbose_json":
                out.update(task="transcribe", language="en", duration=2.5, words=WORDS, segments=[])
            return self.reply(200, out)
        if self.path == "/v1/chat/completions":
            request = json.loads(body)
            self.server.requests.append(("POST", self.path, dict(self.headers), request))
            if request.get("seed", 0) < 0:
                return self.error(400, "seed must be a non-negative integer", "seed")
            out = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "<spk:0> Hello world."},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "transcription": {"language": "en", "duration": 2.5, "text": "Hello world."},
                "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46},
            }
            if request.get("timestamp_granularities"):
                out["transcription"]["words"] = WORDS
            return self.reply(200, out)
        self.reply(404, "")


def write_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
        cls.server.requests = []
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        cls.dir = tempfile.TemporaryDirectory()
        cls.audio = Path(cls.dir.name) / "clip.flac"
        cls.audio.write_bytes(b"fLaC fake audio bytes")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.dir.cleanup()

    def setUp(self):
        self.server.requests.clear()

    def run_client(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = client.main(["--base-url", self.base_url, *argv])
        return status, out.getvalue(), err.getvalue()

    def last(self, method, path):
        matches = [r for r in self.server.requests if r[0] == method and r[1] == path]
        self.assertTrue(matches, f"no {method} {path}")
        return matches[-1]

    def test_model_discovery_prefers_a_transcription_model(self):
        status, out, _ = self.run_client("transcribe", str(self.audio))
        self.assertEqual(status, 0)
        fields = self.last("POST", "/v1/audio/transcriptions")[3]
        self.assertEqual(fields["model"], ["asr-model"])
        self.assertEqual(
            fields["file"], [("clip.flac", b"fLaC fake audio bytes")], "the file is sent unchanged"
        )
        self.assertEqual(out.strip(), "Hello world.")

    def test_transcribe_fields(self):
        status, out, _ = self.run_client(
            "--model",
            "m",
            "transcribe",
            str(self.audio),
            "--language",
            "de",
            "--prompt",
            "Acme",
            "--temperature",
            "0.2",
            "--timestamps",
            "word",
            "--timestamps",
            "segment",
            "--diarize",
            "--max-speakers",
            "3",
            "--show-words",
        )
        self.assertEqual(status, 0)
        fields = self.last("POST", "/v1/audio/transcriptions")[3]
        self.assertEqual(fields["model"], ["m"])
        self.assertEqual(fields["language"], ["de"])
        self.assertEqual(fields["prompt"], ["Acme"])
        self.assertEqual(fields["temperature"], ["0.2"])
        self.assertEqual(
            fields["response_format"], ["verbose_json"], "timestamps imply verbose_json"
        )
        self.assertEqual(fields["timestamp_granularities[]"], ["word", "segment"])
        self.assertEqual(fields["diarization"], ["true"])
        self.assertEqual(fields["max_speaker_count"], ["3"])
        self.assertIn("spk:1 world.", out)
        self.assertNotIn("models", [r[1] for r in self.server.requests], "--model skips discovery")

    def test_text_formats_print_the_body(self):
        status, out, _ = self.run_client(
            "--model", "m", "transcribe", str(self.audio), "--format", "srt"
        )
        self.assertEqual(status, 0)
        self.assertIn("00:00:00,400 --> 00:00:01,300", out)
        output = Path(self.dir.name) / "out.vtt"
        status, out, _ = self.run_client(
            "--model",
            "m",
            "transcribe",
            str(self.audio),
            "--format",
            "vtt",
            "--output",
            str(output),
        )
        self.assertEqual((status, out), (0, ""))
        self.assertIn("Hello world.", output.read_text())

    def test_json_prints_the_whole_response(self):
        status, out, _ = self.run_client(
            "--model", "m", "transcribe", str(self.audio), "--format", "verbose_json", "--json"
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)["words"], WORDS)

    def test_extra_fields(self):
        status, _, _ = self.run_client(
            "--model",
            "m",
            "transcribe",
            str(self.audio),
            "--extra",
            "verbatim=true",
            "--extra",
            'speech_contexts=[{"phrases":["Acme"]}]',
        )
        self.assertEqual(status, 0)
        fields = self.last("POST", "/v1/audio/transcriptions")[3]
        self.assertEqual(fields["verbatim"], ["true"])
        self.assertIn("Acme", fields["speech_contexts"][0])
        status, _, err = self.run_client(
            "--model", "m", "transcribe", str(self.audio), "--extra", "novalue"
        )
        self.assertEqual(status, 2)
        self.assertIn("key=value", err)

    def test_translate(self):
        status, out, _ = self.run_client(
            "--model",
            "m",
            "translate",
            str(self.audio),
            "--language",
            "de",
            "--target-language",
            "fr",
            "--format",
            "text",
        )
        self.assertEqual((status, out.strip()), (0, "Hello world."))
        fields = self.last("POST", "/v1/audio/translations")[3]
        self.assertEqual((fields["language"], fields["target_language"]), (["de"], ["fr"]))

    def test_chat_request_shape(self):
        status, out, _ = self.run_client(
            "--model",
            "m",
            "chat",
            str(self.audio),
            "--instruction",
            "Transcribe with speaker tags.",
            "--language",
            "en",
            "--timestamps",
            "word",
            "--seed",
            "7",
            "--temperature",
            "0.5",
            "--show-words",
        )
        self.assertEqual(status, 0)
        request = self.last("POST", "/v1/chat/completions")[3]
        content = request["messages"][0]["content"]
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(content[0], {"type": "text", "text": "Transcribe with speaker tags."})
        self.assertEqual(content[1]["type"], "input_audio")
        self.assertEqual(
            content[1]["input_audio"]["format"], "flac", "format from the file extension"
        )
        self.assertEqual(content[1]["input_audio"]["data"], "ZkxhQyBmYWtlIGF1ZGlvIGJ5dGVz")
        self.assertEqual((request["seed"], request["temperature"]), (7, 0.5))
        self.assertEqual(request["language"], "en")
        self.assertEqual(request["timestamp_granularities"], ["word"])
        self.assertIn("<spk:0> Hello world.", out)
        self.assertIn("0.400 -", out)

    def test_chat_sends_zero_seed_and_temperature(self):
        status, _, _ = self.run_client(
            "--model", "m", "chat", str(self.audio), "--seed", "0", "--temperature", "0"
        )
        self.assertEqual(status, 0)
        request = self.last("POST", "/v1/chat/completions")[3]
        self.assertEqual((request["seed"], request["temperature"]), (0, 0.0))

    def test_chat_without_instruction_sends_only_audio(self):
        status, _, _ = self.run_client(
            "--model", "m", "chat", str(self.audio), "--audio-format", "wav"
        )
        self.assertEqual(status, 0)
        request = self.last("POST", "/v1/chat/completions")[3]
        self.assertEqual([p["type"] for p in request["messages"][0]["content"]], ["input_audio"])
        self.assertEqual(request["messages"][0]["content"][0]["input_audio"]["format"], "wav")
        self.assertNotIn("seed", request)

    def test_server_errors_are_reported_as_is(self):
        status, _, err = self.run_client("--model", "m", "chat", str(self.audio), "--seed", "-1")
        self.assertEqual(status, 1)
        self.assertIn(
            "HTTP 400 invalid_request: seed must be a non-negative integer (param: seed)", err
        )
        self.assertEqual(
            self.last("POST", "/v1/chat/completions")[3]["seed"], -1, "values go to the server"
        )

    def test_an_endpoint_the_server_does_not_serve(self):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            status = client.main(
                ["--base-url", self.base_url + "/x", "--model", "m", "chat", str(self.audio)]
            )
        err = err.getvalue()
        self.assertEqual(status, 1)
        self.assertIn("error: HTTP 404: Not Found (POST /v1/x/chat/completions)", err)

    def test_usage_and_connection_errors(self):
        status, _, err = self.run_client("--model", "m", "transcribe", "/no/such/file.wav")
        self.assertEqual(status, 2)
        self.assertIn("no such audio file", err)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = client.main(
                [
                    "--base-url",
                    "http://127.0.0.1:9/v1",
                    "--model",
                    "m",
                    "transcribe",
                    str(self.audio),
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("cannot reach the server", err.getvalue())

    def test_batch(self):
        base = Path(self.dir.name)
        (base / "b.wav").write_bytes(b"RIFF fake")
        manifest = base / "manifest.jsonl"
        rows = [
            {"audio_filepath": "clip.flac", "id": "one"},
            {"audio_filepath": str(base / "b.wav"), "language": "xx"},
            {"audio_filepath": "missing.wav"},
            {"audio_filepath": "clip.flac", "language": "fr"},
        ]
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
        output = base / "results.jsonl"
        status, _, err = self.run_client(
            "--model",
            "m",
            "batch",
            str(manifest),
            "--output",
            str(output),
            "--workers",
            "3",
            "--format",
            "verbose_json",
            "--overwrite",
        )
        self.assertEqual(status, 1, "a failed row fails the run")
        results = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r["line"] for r in results], [1, 2, 3, 4], "manifest order")
        self.assertEqual([r["status"] for r in results], ["ok", "error", "error", "ok"])
        self.assertEqual(results[0]["input"]["id"], "one")
        self.assertEqual((results[0]["text"], results[0]["duration"]), ("Hello world.", 2.5))
        self.assertEqual(results[1]["error"]["status"], 400)
        self.assertIn("no such audio file", results[2]["error"]["message"])
        languages = sorted(
            (r[3].get("language") or [None])[0] or ""
            for r in self.server.requests
            if r[1] == "/v1/audio/transcriptions"
        )
        self.assertEqual(languages, ["", "fr", "xx"], "per-row language overrides")
        summary = json.loads(err.strip().splitlines()[-1])
        self.assertEqual(
            (summary["rows"], summary["ok"], summary["error"], summary["audio_seconds"]),
            (4, 2, 2, 5.0),
        )
        self.assertLessEqual(summary["latency_p90_ms"], summary["latency_max_ms"])
        status, _, err = self.run_client(
            "--model", "m", "batch", str(manifest), "--output", str(output)
        )
        self.assertEqual(status, 2, "an existing output is kept without --overwrite")

    def test_batch_chat_mode(self):
        base = Path(self.dir.name)
        manifest = base / "chat.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "clip.flac", "instruction": "Row instruction."}) + "\n"
        )
        output = base / "chat-results.jsonl"
        status, _, _ = self.run_client(
            "--model",
            "m",
            "batch",
            str(manifest),
            "--output",
            str(output),
            "--mode",
            "chat",
            "--overwrite",
            "--include-response",
        )
        self.assertEqual(status, 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["text"], "<spk:0> Hello world.")
        self.assertEqual(result["response"]["transcription"]["duration"], 2.5)
        request = self.last("POST", "/v1/chat/completions")[3]
        self.assertEqual(request["messages"][0]["content"][0]["text"], "Row instruction.")

    def stats_line(self, err):
        lines = [line for line in err.splitlines() if line.startswith("stats: ")]
        self.assertEqual(len(lines), 1, err)
        return lines[0]

    def test_stats(self):
        status, out, err = self.run_client("chat", str(self.audio), "--stats")
        self.assertEqual(
            (status, out.strip()), (0, "<spk:0> Hello world."), "stats stay off stdout"
        )
        line = self.stats_line(err)
        self.assertRegex(
            line, r"^stats: latency \d+ ms \| audio 2\.5 s \| rtfx [\d.]+ \| tokens 6$"
        )
        wav = Path(self.dir.name) / "three.wav"
        write_wav(wav, 3.0)
        status, _, err = self.run_client("transcribe", str(wav), "--stats")
        self.assertEqual(status, 0)
        self.assertIn("| audio 3.0 s |", self.stats_line(err), "the WAV header gives the duration")
        status, _, err = self.run_client("translate", str(self.audio), "--stats")
        self.assertIn("audio n/a", self.stats_line(err), "json: no duration reported, no RTFx")
        self.assertNotIn("rtfx", self.stats_line(err))

    def test_batch_stats(self):
        base = Path(self.dir.name)
        manifest = base / "stats.jsonl"
        manifest.write_text(
            "".join(json.dumps({"audio_filepath": "clip.flac", "n": i}) + "\n" for i in range(3))
        )
        output, stats = base / "stats-results.jsonl", base / "stats.json"
        status, _, err = self.run_client(
            "--model", "m", "batch", str(manifest), "--output", str(output), "--mode", "chat",
            "--warmup", "4", "--stats-json", str(stats), "--overwrite",
        )  # fmt: skip
        self.assertEqual(status, 0)
        chats = [r for r in self.server.requests if r[1] == "/v1/chat/completions"]
        self.assertEqual(len(chats), 7, "4 warm-up requests and 3 rows")
        results = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            [r["input"]["n"] for r in results], [0, 1, 2], "warm-up requests are not written"
        )
        self.assertTrue(all(r["latency_ms"] >= 0 and r["audio_seconds"] == 2.5 for r in results))
        summary = json.loads(stats.read_text())
        self.assertEqual(summary, json.loads(err.strip().splitlines()[-1]))
        self.assertEqual(summary["audio_seconds"], 7.5)
        self.assertGreater(summary["rtfx"], 0)
        self.assertLessEqual(summary["latency_p50_ms"], summary["latency_p99_ms"])
        self.assertLessEqual(summary["latency_p99_ms"], summary["latency_max_ms"])

    def test_bench(self):
        clips = Path(self.dir.name) / "clips"
        clips.mkdir(exist_ok=True)
        for name, seconds in (("a.wav", 2.0), ("b.wav", 4.0)):
            write_wav(clips / name, seconds)
        (clips / "notes.txt").write_text("not audio")
        out = Path(self.dir.name) / "bench"
        status, stdout, _ = self.run_client(
            "--model", "m", "bench", str(clips), "--out", str(out), "--conc", "1", "2", "--min-requests", "3",
            "--warmup", "2",
        )  # fmt: skip
        self.assertEqual(status, 0)
        chats = [r for r in self.server.requests if r[1] == "/v1/chat/completions"]
        self.assertEqual(len(chats), 2 + 3 + 4, "warm-up, then max(2 N, --min-requests) per level")
        self.assertTrue(all("stream" not in r[3] for r in chats), "responses are not streamed")
        rows = [line.split("\t") for line in (out / "points.tsv").read_text().splitlines()]
        self.assertEqual(rows[0][:3], ["conc", "requests", "errors"])
        point = dict(zip(rows[0], rows[2]))
        self.assertEqual((point["conc"], point["requests"], point["errors"]), ("2", "4", "0"))
        self.assertEqual(point["audio_s"], "12.0", "clips round robin across levels: b, a, b, a")
        self.assertEqual(json.loads((out / "c1.json").read_text())["requests"], 3)
        self.assertIn("latency_p50_ms", stdout.splitlines()[0])
        for mode, path in (
            ("chat", "/v1/chat/completions"),
            ("transcribe", "/v1/audio/transcriptions"),
        ):
            self.server.requests.clear()
            other = Path(self.dir.name) / f"bench-{mode}"
            status, _, _ = self.run_client(
                "--model", "m", "bench", str(clips), "--out", str(other), "--conc", "2", "--min-requests", "2",
                "--warmup", "0", "--mode", mode,
            )  # fmt: skip
            self.assertEqual(status, 0)
            sent = [r for r in self.server.requests if r[1] == path]
            self.assertEqual(len(sent), 4, mode)
            point = json.loads((other / "c2.json").read_text())
            self.assertEqual((point["errors"], point["audio_s"]), (0, 12.0), mode)
            self.assertNotIn("first_token_p50_ms", point)
        status, _, err = self.run_client(
            "--model", "m", "bench", str(Path(self.dir.name) / "none"), "--out", str(out)
        )
        self.assertEqual(status, 2)

    def test_invalid_manifest(self):
        manifest = Path(self.dir.name) / "bad.jsonl"
        manifest.write_text('{"audio_filepath": "clip.flac"}\nnot json\n')
        status, _, err = self.run_client(
            "--model",
            "m",
            "batch",
            str(manifest),
            "--output",
            str(Path(self.dir.name) / "bad-out.jsonl"),
            "--overwrite",
        )
        self.assertEqual(status, 2)
        self.assertIn("bad.jsonl:2: invalid JSON", err)


if __name__ == "__main__":
    unittest.main()
