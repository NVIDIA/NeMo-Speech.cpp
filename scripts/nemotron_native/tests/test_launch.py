# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib.util
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "native_launch_affinity", Path(__file__).parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def options(native=None, frontend=None, port=None):
    return SimpleNamespace(native_cpus=native, frontend_cpus=frontend, http_port=port)


class CpuConfigurationTests(unittest.TestCase):
    def test_defaults_keep_commands_and_do_not_require_taskset_or_linux(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform), mock.patch.object(
                launch.sys, "platform", platform
            ), mock.patch.object(
                launch.os, "sched_getaffinity", return_value={12, 13}, create=True
            ), mock.patch.object(
                launch.shutil, "which"
            ) as which:
                taskset, configuration = launch.cpu_configuration(options(port=8000))
                self.assertIsNone(taskset)
                which.assert_not_called()
                for role in ("worker", "bridge", "frontend"):
                    command = ["program", "--option"]
                    self.assertIs(
                        launch.cpu_command(command, role, taskset, configuration), command
                    )
                    self.assertIsNone(configuration["roles"][role]["requested_cpus"])

    def test_cpu_ranges_are_validated_and_shared_sets_are_allowed(self):
        with mock.patch.object(launch.sys, "platform", "linux"), mock.patch.object(
            launch.os, "sched_getaffinity", return_value={12, 13, 36, 37}, create=True
        ), mock.patch.object(launch.shutil, "which", return_value="/usr/bin/taskset"):
            taskset, configuration = launch.cpu_configuration(
                options("12-13, 36,12", "12-13,36-37", 8000)
            )
            for role in ("worker", "bridge"):
                self.assertEqual(configuration["roles"][role]["requested_cpus"], [12, 13, 36])
                self.assertEqual(
                    launch.cpu_command(["program"], role, taskset, configuration),
                    ["/usr/bin/taskset", "--cpu-list", "12,13,36", "program"],
                )
            self.assertEqual(configuration["roles"]["frontend"]["requested_cpus"], [12, 13, 36, 37])
            for invalid in (
                "",
                "-1",
                "12,,13",
                "13-12",
                "12.0",
                "12:13",
                "14",
                "12-36",
                "999999999",
            ):
                with self.subTest(value=invalid), self.assertRaises(ValueError):
                    launch.cpu_configuration(options(native=invalid))

    def test_each_control_only_prefixes_its_selected_roles(self):
        with mock.patch.object(launch.sys, "platform", "linux"), mock.patch.object(
            launch.os, "sched_getaffinity", return_value={12, 13}, create=True
        ), mock.patch.object(launch.shutil, "which", return_value="/usr/bin/taskset"):
            for args, selected in (
                (options(native="12", port=8000), {"worker", "bridge"}),
                (options(frontend="13", port=8000), {"frontend"}),
            ):
                taskset, configuration = launch.cpu_configuration(args)
                for role in ("worker", "bridge", "frontend"):
                    with self.subTest(role=role, selected=selected):
                        command = ["program"]
                        result = launch.cpu_command(command, role, taskset, configuration)
                        self.assertEqual(result != command, role in selected)

    def test_frontend_selection_requires_frontend_and_explicit_controls_require_tools(self):
        with self.assertRaisesRegex(ValueError, "frontend-cpus requires http-port"):
            launch.cpu_configuration(options(frontend="12"))
        with mock.patch.object(launch.sys, "platform", "darwin"), self.assertRaisesRegex(
            ValueError, "require Linux"
        ):
            launch.cpu_configuration(options(native="12"))
        with mock.patch.object(launch.sys, "platform", "linux"), mock.patch.object(
            launch.os, "sched_getaffinity", return_value={12}, create=True
        ), mock.patch.object(launch.shutil, "which", return_value=None), self.assertRaisesRegex(
            ValueError, "require taskset"
        ):
            launch.cpu_configuration(options(native="12"))


class CpuLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_cpu_is_rejected_before_any_subprocess(self):
        with mock.patch.object(launch.sys, "platform", "linux"), mock.patch.object(
            launch.os, "sched_getaffinity", return_value={12}, create=True
        ), mock.patch.object(
            launch.shutil, "which", return_value="/usr/bin/taskset"
        ), mock.patch.object(
            launch.asyncio, "create_subprocess_exec"
        ) as spawn:
            with self.assertRaisesRegex(ValueError, "launcher CPU affinity"):
                await launch.run(options(native="13"))
            spawn.assert_not_called()

    @unittest.skipUnless(
        sys.platform == "linux" and hasattr(os, "sched_getaffinity") and shutil.which("taskset"),
        "Linux taskset required",
    )
    async def test_role_affinity_precedes_child_threads_and_is_recorded_in_both_receipts(self):
        available = sorted(os.sched_getaffinity(0))
        native_cpus = available[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "PACKAGE.json").write_text("{}")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
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
                    "--http-port",
                    str(port),
                    "--frontend-binary",
                    "/test/frontend",
                    "--native-cpus",
                    ",".join(map(str, native_cpus)),
                    "--frontend-cpus",
                    ",".join(map(str, available)),
                    "--startup-timeout",
                    "5",
                    "--shutdown-timeout",
                    "1",
                ]
            )

            def child_code(path, listen=""):
                return f"""import json,os,pathlib,socket,threading,time
observed={{'main':sorted(os.sched_getaffinity(0))}}
thread=threading.Thread(target=lambda: observed.update(thread=sorted(os.sched_getaffinity(0))))
thread.start()
thread.join()
path=pathlib.Path({str(path)!r})
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(observed))
{listen}
time.sleep(60)
"""

            worker_path = args.receipts / "worker/NATIVE_READY.json"
            bridge_path = root / "bridge-observed.json"
            frontend_path = root / "frontend-observed.json"
            worker_command = [sys.executable, "-c", child_code(worker_path)]
            bridge_command = [
                sys.executable,
                "-c",
                child_code(
                    bridge_path,
                    f"s=socket.socket(socket.AF_UNIX); s.bind({str(args.listen)!r}); s.listen(1)",
                ),
            ]
            frontend_code = child_code(
                frontend_path, f"s=socket.socket(); s.bind(('127.0.0.1',{port})); s.listen(1)"
            )
            original_spawn = asyncio.create_subprocess_exec
            spawned = []

            async def spawn(*command, **kwargs):
                spawned.append(command)
                if "/test/frontend" in command:
                    index = command.index("/test/frontend")
                    command = (*command[:index], sys.executable, "-c", frontend_code)
                return await original_spawn(*command, **kwargs)

            callbacks = {}
            loop = asyncio.get_running_loop()
            with mock.patch.object(
                launch, "commands", return_value=(worker_command, bridge_command)
            ), mock.patch.object(
                launch.asyncio, "create_subprocess_exec", side_effect=spawn
            ), mock.patch.object(
                loop,
                "add_signal_handler",
                side_effect=lambda number, callback: callbacks.setdefault(number, callback),
            ):
                task = asyncio.create_task(launch.run(args))
                try:

                    async def ready():
                        while not (args.receipts / "LAUNCH.json").exists():
                            if task.done():
                                task.result()
                            await asyncio.sleep(0.01)

                    await asyncio.wait_for(ready(), 5)
                finally:
                    if signal.SIGTERM in callbacks:
                        callbacks[signal.SIGTERM]()
                    await asyncio.wait_for(task, 5)
            for role, path, expected in (
                ("worker", worker_path, native_cpus),
                ("bridge", bridge_path, native_cpus),
                ("frontend", frontend_path, available),
            ):
                self.assertEqual(
                    json.loads(path.read_text()), {"main": expected, "thread": expected}
                )
                for name in ("LAUNCH.json", "EXIT.json"):
                    report = json.loads((args.receipts / name).read_text())
                    record = report["cpu_affinity"]["roles"][role]
                    self.assertEqual(record["requested_cpus"], expected)
                    self.assertEqual(record["effective_cpus"], expected)
                    self.assertGreater(record["pid"], 0)
            self.assertEqual(len(spawned), 3)
            self.assertTrue(all(command[1] == "--cpu-list" for command in spawned))


if __name__ == "__main__":
    unittest.main()
