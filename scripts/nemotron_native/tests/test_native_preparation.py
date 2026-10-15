# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes as C
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from nemotron_cpp_bridge.native_preparation import FrontendPreparation, NativePreparation


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Library:
    def __init__(self):
        self.buffers = {}
        self.freed = []
        self.destroyed = []
        self.inputs = []
        self.result = {"text": "Hello", "tokens": [1, 2, 131073]}
        self.error = None
        self.status = 0
        self.started = None
        self.release = None
        self.nemotron_text_create = Function(self.create)
        self.nemotron_text_prepare = Function(self.prepare)
        self.nemotron_text_free = Function(self.free)
        self.nemotron_text_destroy = Function(self.destroy)

    def assign(self, pointer, value):
        C.cast(pointer, C.POINTER(C.c_void_p))[0] = C.c_void_p(value)

    def allocate(self, pointer, data):
        buffer = C.create_string_buffer(data)
        address = C.addressof(buffer)
        self.buffers[address] = buffer
        self.assign(pointer, address)

    def create(self, config, tokenizer, handle, error):
        self.assign(handle, 123)
        return 0

    def prepare(self, handle, data, length, normalize, output, error):
        self.inputs.append((C.string_at(data, length), bool(normalize)))
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(5):
            raise TimeoutError("test did not release native preparation")
        if self.result is not None:
            data = (
                self.result if isinstance(self.result, bytes) else json.dumps(self.result).encode()
            )
            self.allocate(output, data)
        if self.error is not None:
            self.allocate(error, self.error.encode())
        return self.status

    def free(self, pointer):
        self.freed.append(pointer.value)
        del self.buffers[pointer.value]

    def destroy(self, handle):
        self.destroyed.append(handle.value)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.library_path = Path(self.temp.name, "preparation.so")
        self.library_path.touch()
        self.library = Library()
        self.patch = mock.patch(
            "nemotron_cpp_bridge.native_preparation.C.CDLL", return_value=self.library
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.preparer = NativePreparation(self.library_path, "tokenizer.json")
        self.addCleanup(self.preparer.close)

    def test_owned_output_is_freed_and_input_trimming_is_preserved(self):
        self.assertEqual(
            self.preparer.prepare_tokens(" \tHello\n", False), ("Hello", [1, 2, 131073])
        )
        self.assertEqual(self.library.inputs, [(b"Hello", False)])
        self.assertFalse(self.library.buffers)
        self.assertEqual(len(self.library.freed), 1)

    def test_bad_request_types_do_not_call_native_code(self):
        for text, normalize in ((None, True), (123, True), ("hello", "yes"), ("hello", 1)):
            with self.subTest(text=text, normalize=normalize), self.assertRaises(ValueError):
                self.preparer.prepare_tokens(text, normalize)
        self.assertFalse(self.library.inputs)

    def test_native_errors_and_malformed_results_release_every_allocation(self):
        for status, exception in ((1, ValueError), (2, RuntimeError)):
            self.library.status, self.library.error = status, "rejected"
            with self.subTest(status=status), self.assertRaisesRegex(exception, "rejected"):
                self.preparer.prepare_tokens("hello")
            self.assertFalse(self.library.buffers)
            self.assertEqual(self.preparer._active, 0)
        self.library.status, self.library.error, self.library.result = 0, None, b"not json"
        with self.assertRaises(ValueError):
            self.preparer.prepare_tokens("hello")
        self.assertFalse(self.library.buffers)
        self.assertEqual(self.preparer._active, 0)
        self.library.result = None
        with self.assertRaisesRegex(RuntimeError, "empty result"):
            self.preparer.prepare_tokens("hello")

    def test_cancel_and_close_preserve_running_call_ownership(self):
        self.library.started, self.library.release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as workers, ThreadPoolExecutor(
            max_workers=1
        ) as closer:
            running = workers.submit(self.preparer.prepare_tokens, "running")
            self.assertTrue(self.library.started.wait(3))
            queued = workers.submit(self.preparer.prepare_tokens, "queued")
            self.assertTrue(queued.cancel())
            self.assertFalse(running.cancel())
            closed = closer.submit(self.preparer.close)
            try:
                with self.preparer._condition:
                    self.assertTrue(
                        self.preparer._condition.wait_for(lambda: self.preparer._closed, 3)
                    )
                self.assertFalse(closed.done())
                self.assertFalse(self.library.destroyed)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    self.preparer.prepare_tokens("late")
            finally:
                self.library.release.set()
            self.assertEqual(running.result(3), ("Hello", [1, 2, 131073]))
            closed.result(3)
        self.preparer.close()
        self.assertEqual(self.library.destroyed, [123])
        self.assertEqual(self.library.inputs, [(b"running", True)])
        self.assertFalse(self.library.buffers)

    def test_invalid_configuration_is_rejected(self):
        config = Path(self.temp.name, "config.json")
        for wrapper in (NativePreparation, FrontendPreparation):
            for value in (None, False, [], "wrong", 1):
                config.write_text(json.dumps(value))
                with self.subTest(wrapper=wrapper.__name__, value=value), self.assertRaisesRegex(
                    ValueError, "normalization configuration must be an object"
                ):
                    wrapper(self.library_path, "tokenizer.json", config)

    def test_absent_configuration_and_explicit_disabled_configuration_remain_supported(self):
        config = Path(self.temp.name, "config.json")
        config.write_text(json.dumps({"enabled": False}))
        for wrapper in (NativePreparation, FrontendPreparation):
            for path in (None, config):
                with self.subTest(wrapper=wrapper.__name__, config=path):
                    preparer = wrapper(self.library_path, "tokenizer.json", path)
                    try:
                        self.assertIs(preparer.enabled, False)
                        if isinstance(preparer, FrontendPreparation):
                            self.assertEqual(
                                json.loads(preparer.metadata["normalization_config"]),
                                None if path is None else {"enabled": False},
                            )
                    finally:
                        preparer.close()


NATIVE_LIBRARY = os.environ.get("NEMOTRON_NATIVE_PREPARATION_TEST_LIBRARY")
TOKENIZER = os.environ.get("NEMOTRON_NATIVE_PREPARATION_TEST_TOKENIZER")
NORMALIZATION = os.environ.get("NEMOTRON_NORMALIZATION_CONFIG")
CORPUS = os.environ.get("NEMOTRON_NATIVE_PREPARATION_TEST_CORPUS")


@unittest.skipUnless(NATIVE_LIBRARY and TOKENIZER and NORMALIZATION, "native fixtures not provided")
class NativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nemotron_cpp_bridge.normalization import TextPreparer
        from tokenizers import Tokenizer

        cls.reference = TextPreparer(NORMALIZATION)
        cls.tokenizer = Tokenizer.from_file(TOKENIZER)
        cls.native = NativePreparation(NATIVE_LIBRARY, TOKENIZER, NORMALIZATION)
        cls.texts = [
            "  Hello world  ",
            "I paid $12.50 for 3 books.",
            "Call me at 5 p.m.",
            "Dr. Smith can't arrive until 3:30 p.m.",
            "The year is 2026.",
            "Call 1-800-555-0100.",
            "Cafe\u0301 café naïve façade",
            "中文 日本語 한국어",
            "😀🧑🏽‍🚀",
            "line one\nline two\tthree\r\nfour",
            "\x1c\x1d\x1e\x1f\u0085\u00a0Hello!\u00a0\x1f",
            "Keep spaces , and punctuation !",
            "",
            " \t\n",
            "word " * 513,
        ]
        if CORPUS:
            cls.texts.extend(
                line.split("\t", 1)[1] for line in Path(CORPUS).read_text().splitlines()
            )

    @classmethod
    def tearDownClass(cls):
        cls.native.close()
        cls.reference.close()

    @classmethod
    def expected(cls, text, normalize):
        normalized = cls.reference.prepare(text, "en-US", normalize)
        tokens = cls.tokenizer.encode_batch([normalized], add_special_tokens=False)[0].ids + [
            131073
        ]
        if not 1 <= len(tokens) <= 512 or any(not 0 <= token < 131075 for token in tokens):
            raise ValueError("token budget or vocabulary")
        return normalized, tokens

    def test_exact_corpus_and_edge_cases_with_and_without_normalization(self):
        for text in self.texts:
            for normalize in (False, True):
                with self.subTest(text=text[:60], normalize=normalize):
                    try:
                        expected = self.expected(text, normalize)
                    except (ValueError, RuntimeError) as error:
                        with self.assertRaises(type(error)):
                            self.native.prepare_tokens(text, normalize)
                    else:
                        self.assertEqual(self.native.prepare_tokens(text, normalize), expected)

    def test_parallel_preparation_matches_reference(self):
        texts = self.texts[:5] * 26
        expected = [self.expected(text, True) for text in texts]
        with ThreadPoolExecutor(max_workers=32) as executor:
            actual = list(executor.map(self.native.prepare_tokens, texts))
        self.assertEqual(actual, expected)

    def test_invalid_native_configuration_is_rejected(self):
        base = json.loads(Path(NORMALIZATION).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "config.json")
            for change in (
                {"unknown": True},
                {"enabled": "yes"},
                {"pool_size": 0},
                {"pool_size": -1},
                {"language_code": "de-DE"},
                {"max_input_bytes": 0},
                {"normalizer_config": "../outside"},
                {"normalizer_root": base["native_library_root"]},
            ):
                path.write_text(json.dumps({**base, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    NativePreparation(NATIVE_LIBRARY, TOKENIZER, path)


if __name__ == "__main__":
    unittest.main()
