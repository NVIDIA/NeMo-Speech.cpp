# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "native_launch", Path(__file__).parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class LauncherTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_shutdown_and_forced_worker_cleanup_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "PACKAGE.json").write_text("{}")
            args = launch.parse_args(
                [
                    "--package",
                    str(package),
                    "--model",
                    str(root),
                    "--width",
                    "1",
                    "--listen",
                    str(root / "bridge.sock"),
                    "--receipts",
                    str(root / "receipts"),
                    "--startup-timeout",
                    "3",
                    "--shutdown-timeout",
                    "0.05",
                ]
            )
            worker_code = (
                "import pathlib,signal,time; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"p=pathlib.Path({str(args.receipts / 'worker/NATIVE_READY.json')!r}); "
                "p.parent.mkdir(parents=True); p.write_text('{}'); time.sleep(60)"
            )
            bridge_code = (
                "import socket,time; s=socket.socket(socket.AF_UNIX); "
                f"s.bind({str(args.listen)!r}); s.listen(1); time.sleep(60)"
            )
            commands = ([sys.executable, "-c", worker_code], [sys.executable, "-c", bridge_code])
            callbacks = {}
            loop = asyncio.get_running_loop()
            with mock.patch.object(launch, "commands", return_value=commands), mock.patch.object(
                loop,
                "add_signal_handler",
                side_effect=lambda number, callback: callbacks.setdefault(number, callback),
            ):
                task = asyncio.create_task(launch.run(args))

                async def observe():
                    while not (args.receipts / "LAUNCH.json").exists():
                        if task.done():
                            task.result()
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(observe(), 3)
                callbacks[signal.SIGTERM]()
                await asyncio.wait_for(task, 3)
            report = json.loads((args.receipts / "EXIT.json").read_text())
            self.assertEqual(report["status"], "STOPPED")
            self.assertEqual(report["exit_codes"], [-signal.SIGKILL, -signal.SIGTERM])

    async def test_startup_child_failure_stops_remaining_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "PACKAGE.json").write_text("{}")
            args = launch.parse_args(
                [
                    "--package",
                    str(root),
                    "--model",
                    str(root),
                    "--width",
                    "1",
                    "--listen",
                    str(root / "bridge.sock"),
                    "--receipts",
                    str(root / "receipts"),
                    "--startup-timeout",
                    "3",
                    "--shutdown-timeout",
                    "0.05",
                ]
            )
            loop = asyncio.get_running_loop()
            with mock.patch.object(
                launch,
                "commands",
                return_value=(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    [sys.executable, "-c", "pass"],
                ),
            ), mock.patch.object(loop, "add_signal_handler"):
                with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                    await asyncio.wait_for(launch.run(args), 3)
            report = json.loads((args.receipts / "EXIT.json").read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["exit_codes"], [7])


if __name__ == "__main__":
    unittest.main()
