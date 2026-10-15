# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import ctypes as C
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nemotron_cpp_bridge.normalization import TextPreparer
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser()
    for name in ("library", "model", "config", "corpus", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    lib = C.CDLL(str(args.library))
    lib.nemotron_text_create.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    lib.nemotron_text_create.restype = C.c_int
    lib.nemotron_text_prepare.argtypes = [
        C.c_void_p,
        C.c_char_p,
        C.c_size_t,
        C.c_uint8,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    lib.nemotron_text_prepare.restype = C.c_int
    lib.nemotron_text_free.argtypes = [C.c_void_p]
    lib.nemotron_text_free.restype = None
    lib.nemotron_text_destroy.argtypes = [C.c_void_p]
    lib.nemotron_text_destroy.restype = None
    context, error = C.c_void_p(), C.c_void_p()
    code = lib.nemotron_text_create(
        args.config.read_bytes(),
        str(args.model / "tokenizer.json").encode(),
        C.byref(context),
        C.byref(error),
    )
    if code:
        message = C.string_at(error).decode()
        lib.nemotron_text_free(error)
        raise RuntimeError(message)
    preparer = TextPreparer(args.config)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    def native(text, apply=True):
        data = text.encode()
        out, error = C.c_void_p(), C.c_void_p()
        code = lib.nemotron_text_prepare(
            context, data, len(data), apply, C.byref(out), C.byref(error)
        )
        try:
            if code:
                message = C.string_at(error).decode()
                raise (ValueError if code == 1 else RuntimeError)(message)
            value = json.loads(C.string_at(out))
            return value["text"], value["tokens"]
        finally:
            lib.nemotron_text_free(out)
            lib.nemotron_text_free(error)

    def reference(text, apply=True):
        text = preparer.prepare(text, "en-US", apply)
        ids = tokenizer.encode_batch([text], add_special_tokens=False)[0].ids + [131073]
        if len(ids) > 512 or any(x >= 131075 for x in ids):
            raise ValueError("invalid token IDs")
        return text, ids

    corpus = [line.split("\t", 1)[1] for line in args.corpus.read_text().splitlines()]
    edges = [
        "  Hello world.  ",
        "\tHello\n",
        "The cost is $12.50.",
        "Dr. Smith arrived at 12:30 p.m.",
        "Call 415-555-0123.",
        "On 01/02/2025, 3.14 percent rose.",
        "Version 2.1 and B512.",
        "Café naïve résumé.",
        "What ? Yes ! Fine , thanks .",
        "नमस्ते ।",
        "Hello 😀 world.",
        "a\0b",
        "\u001cHello\u001f",
        "\u0085Hello\u00a0",
        "\u200bHello\u200b",
        "<speak>hello</speak>",
        "one\ntwo\nthree",
        "",
        "    ",
        "x " * 600,
        "x" * 65537,
    ]
    records = []
    try:
        for text in corpus + edges:
            for apply in (False, True):
                outcomes = []
                for fn in (reference, native):
                    try:
                        outcomes.append(("ok", fn(text, apply)))
                    except Exception as e:
                        outcomes.append((type(e).__name__, None))
                if outcomes[0] != outcomes[1]:
                    raise AssertionError((repr(text[:100]), apply, outcomes))
                records.append(
                    {
                        "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "normalize": apply,
                        "status": outcomes[0][0],
                    }
                )
        expected = [reference(text) for text in corpus]
        timings = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            for name, fn in [
                ("python", reference),
                ("native", native),
                ("native", native),
                ("python", reference),
            ]:
                list(pool.map(fn, [corpus[i % len(corpus)] for i in range(32)]))
                started = time.perf_counter()
                outputs = list(pool.map(fn, [corpus[i % len(corpus)] for i in range(512)]))
                elapsed = (time.perf_counter() - started) * 1000
                assert outputs == [expected[i % len(corpus)] for i in range(512)]
                timings.append(
                    {
                        "implementation": name,
                        "requests": 512,
                        "threads": 16,
                        "wall_ms": elapsed,
                    }
                )
        result = {
            "status": "PASS",
            "cases": len(records),
            "records": records,
            "concurrent_requests": 2048,
            "timings": timings,
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    finally:
        preparer.close()
        lib.nemotron_text_destroy(context)


if __name__ == "__main__":
    main()
