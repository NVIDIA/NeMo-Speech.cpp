# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "normalization",
    Path(__file__).resolve().parents[1] / "python/nemotron_cpp_bridge/normalization.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparationTests(unittest.TestCase):
    def test_bypass_trims_without_changing_punctuation(self):
        preparer = MODULE.TextPreparer()
        self.assertFalse(preparer.enabled)
        self.assertEqual(preparer.prepare(" \tKeep 2 apples\n", "en-US"), "Keep 2 apples")
        self.assertEqual(preparer.prepare("Numbers .", "EN-us", False), "Numbers .")
        preparer.close()
        preparer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            preparer.prepare("Hello", "en-US")

    def test_rejects_invalid_request(self):
        preparer = MODULE.TextPreparer()
        for text, language, normalize in [
            (" ", "en-US", True),
            ("ok", "de-DE", True),
            ("ok", "en-US", "true"),
            (None, "en-US", True),
            ("x" * 65537, "en-US", True),
        ]:
            with self.subTest(text=str(text)[:10], language=language, normalize=normalize):
                with self.assertRaises(ValueError):
                    preparer.prepare(text, language, normalize)
        preparer.close()

    def test_pool_does_not_share_handles_and_close_waits(self):
        preparer = MODULE.TextPreparer()
        preparer.enabled = True
        preparer._handles = [1, 2]
        entered = threading.Barrier(3)
        release = threading.Event()
        destroyed = []
        preparer._api = type("Api", (), {"destroy": lambda self, h: destroyed.append(h)})()

        def process(handle, text):
            entered.wait(timeout=5)
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release normalizer")
            return handle, text

        preparer._process = process
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(preparer.prepare, "one", "en-US")
            second = executor.submit(preparer.prepare, "two", "en-US")
            entered.wait(timeout=5)
            closing = executor.submit(preparer.close)
            self.assertFalse(closing.done())
            self.assertFalse(destroyed)
            release.set()
            self.assertNotEqual(first.result(timeout=5)[0], second.result(timeout=5)[0])
            closing.result(timeout=5)
        self.assertEqual(sorted(destroyed), [1, 2])


@unittest.skipUnless(
    os.environ.get("NEMOTRON_NORMALIZATION_CONFIG"), "native normalizer config not provided"
)
class NativePreparationTests(unittest.TestCase):
    def test_native_abi_and_parallel_replay(self):
        preparer = MODULE.TextPreparer(os.environ["NEMOTRON_NORMALIZATION_CONFIG"])
        try:
            self.assertTrue(preparer.enabled)
            texts = [
                "  Hello world  ",
                "I paid $12.50 for 3 books.",
                "Call me at 5 p.m.",
                "The year is 2026.",
            ]
            expected = [preparer.prepare(text, "en-US") for text in texts]
            self.assertEqual(expected[0], "Hello world.")
            self.assertNotIn("$", expected[1])
            self.assertNotIn("12.50", expected[1])
            self.assertEqual(preparer.prepare("  Keep 2 apples  ", "en-US", False), "Keep 2 apples")
            with ThreadPoolExecutor(max_workers=32) as executor:
                actual = list(
                    executor.map(
                        lambda index: preparer.prepare(texts[index % 4], "en-US"), range(128)
                    )
                )
            self.assertEqual(actual, expected * 32)
        finally:
            preparer.close()


if __name__ == "__main__":
    unittest.main()
