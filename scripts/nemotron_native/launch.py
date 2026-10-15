# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Supervise native TTS worker and C++ bridge")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-seal", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("fixed_cohort", "fixed_request", "inflight"), default="inflight"
    )
    parser.add_argument("--listen", type=Path, required=True)
    parser.add_argument("--native-worker-socket", type=Path)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--normalization-config", type=Path)
    parser.add_argument("--exercise-lifecycle", action="store_true")
    parser.add_argument("--http-port", type=int)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-threads", type=int)
    parser.add_argument("--frontend-binary", default="nemo-speech")
    parser.add_argument("--native-cpus")
    parser.add_argument("--frontend-cpus")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0 or args.shutdown_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.http_port is not None and not 1 <= args.http_port <= 65535:
        parser.error("http-port must be in 1..65535")
    if args.http_threads is not None and args.http_threads <= 0:
        parser.error("http-threads must be positive")
    try:
        cpu_configuration(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.package = args.package.resolve()
    args.model = args.model.resolve()
    args.model_seal = (args.model_seal or args.model / "model-seal.json").resolve()
    args.listen = args.listen.resolve()
    args.native_worker_socket = (
        args.native_worker_socket or args.listen.with_name(args.listen.name + ".worker")
    ).resolve()
    args.receipts = args.receipts.resolve()
    return args


def parse_cpus(value, available):
    selected = set()
    for part in value.split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", part.strip())
        if match is None:
            raise ValueError("CPU sets must use comma-separated CPU IDs or inclusive ranges")
        first = int(match[1])
        last = int(match[2]) if match[2] else first
        if first > last or last > max(available, default=-1):
            raise ValueError("requested CPUs must be within the launcher CPU affinity")
        selected.update(range(first, last + 1))
    if not selected or not selected.issubset(available):
        raise ValueError("requested CPUs must be within the launcher CPU affinity")
    return sorted(selected)


def cpu_configuration(args):
    native = args.native_cpus
    frontend = args.frontend_cpus
    if frontend is not None and args.http_port is None:
        raise ValueError("frontend-cpus requires http-port")
    supported = sys.platform == "linux" and hasattr(os, "sched_getaffinity")
    requested = native is not None or frontend is not None
    if requested and not supported:
        raise ValueError("CPU role controls require Linux CPU affinity support")
    available = set(os.sched_getaffinity(0)) if supported else None
    taskset = shutil.which("taskset") if requested else None
    if requested and taskset is None:
        raise ValueError("CPU role controls require taskset in PATH")
    roles = {"worker": native, "bridge": native}
    if args.http_port is not None:
        roles["frontend"] = frontend
    configuration = {
        "available_cpus": sorted(available) if available is not None else None,
        "roles": {
            role: {
                "requested_cpus": parse_cpus(value, available) if value is not None else None,
                "effective_cpus": None,
            }
            for role, value in roles.items()
        },
    }
    return taskset, configuration


def cpu_command(command, role, taskset, configuration):
    cpus = configuration["roles"][role]["requested_cpus"]
    return [taskset, "--cpu-list", ",".join(map(str, cpus)), *command] if cpus else command


def record_cpu_affinity(configuration, role, process):
    record = configuration["roles"][role]
    record["pid"] = process.pid
    if configuration["available_cpus"] is not None:
        record["effective_cpus"] = sorted(os.sched_getaffinity(process.pid))
        if (
            record["requested_cpus"] is not None
            and record["effective_cpus"] != record["requested_cpus"]
        ):
            raise RuntimeError(role + " CPU affinity differs from its requested CPUs")


def commands(args):
    programs = args.package / "programs"
    selected = programs / f"b{args.width}"
    serving = selected / "serving.json"
    plan = json.loads(serving.read_text())
    if (
        plan["physical_width"] != args.width
        or plan["host"].get("batching_mode", "fixed_cohort") != args.mode
    ):
        raise ValueError("width or mode differs from the assembled package")
    worker = [
        sys.executable,
        "-B",
        str(REPOSITORY / "deploy/nemotron_native/serve.py"),
        "--qualification-only",
        "--plan",
        str(serving),
        "--neural-plan",
        str(selected / "neural.json"),
        "--readiness",
        str(programs / "b1/neural.json"),
        "--readiness-artifacts",
        str(programs / "b1/artifacts"),
        "--artifacts",
        str(selected / "artifacts"),
        "--model",
        str(args.model),
        "--model-seal",
        str(args.model_seal),
        "--source-seal",
        str(args.package / "source-seal.json"),
        "--receipts",
        str(args.receipts / "worker"),
        "--socket",
        str(args.native_worker_socket),
    ]
    bridge = [
        sys.executable,
        "-B",
        "-m",
        "nemotron_cpp_bridge",
        "--native-worker-socket",
        str(args.native_worker_socket),
        "--manifest",
        str(serving),
        "--model",
        str(args.model),
        "--model-seal",
        str(args.model_seal),
        "--listen",
        str(args.listen),
        "--width",
        str(args.width),
        "--mode",
        args.mode,
    ]
    if args.exercise_lifecycle:
        worker.append("--exercise-lifecycle")
    if args.normalization_config:
        bridge.extend(["--normalization-config", str(args.normalization_config.resolve())])
    return worker, bridge


async def wait_ready(path, processes, timeout, stopping):
    async def observe():
        while not path.exists():
            if stopping.is_set():
                raise InterruptedError("startup interrupted")
            if any(process.returncode is not None for process in processes):
                raise RuntimeError("a supervised process exited during startup")
            await asyncio.sleep(0.1)

    await asyncio.wait_for(observe(), timeout)


async def wait_http(host, port, processes, timeout, stopping):
    connect_host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host

    async def observe():
        while True:
            if stopping.is_set():
                raise InterruptedError("startup interrupted")
            if any(process.returncode is not None for process in processes):
                raise RuntimeError("a supervised process exited during HTTP startup")
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection(connect_host, port), 1)
                writer.close()
                await writer.wait_closed()
                return
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.1)

    await asyncio.wait_for(observe(), timeout)


async def run(args):
    taskset, affinity = cpu_configuration(args)
    worker_command, bridge_command = commands(args)
    worker_command = cpu_command(worker_command, "worker", taskset, affinity)
    bridge_command = cpu_command(bridge_command, "bridge", taskset, affinity)
    if args.listen == args.native_worker_socket:
        raise ValueError("worker and bridge sockets must differ")
    for path in (args.listen, args.native_worker_socket):
        if path.exists() or path.is_symlink():
            raise ValueError("refusing to replace existing socket: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.receipts.exists() and any(args.receipts.iterdir()):
        raise ValueError("receipts directory must be new or empty")
    args.receipts.mkdir(parents=True, exist_ok=True)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(REPOSITORY / "python")
        + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        NVIDIA_TF32_OVERRIDE="0",
        CUDA_MODULE_LOADING="LAZY",
    )
    package = json.loads((args.package / "PACKAGE.json").read_text())
    profile = package.get("deployment_profiles", {}).get(str(args.width), {})
    for key, value in profile.get("environment", {}).items():
        if value is not None:
            environment.setdefault(key, value)
    processes = []
    logs = []
    tasks = []
    report = {
        "status": "STARTING",
        "width": args.width,
        "mode": args.mode,
        "cpu_affinity": affinity,
    }
    try:
        log = (args.receipts / "worker.log").open("wb")
        logs.append(log)
        worker = await asyncio.create_subprocess_exec(
            *worker_command,
            env=environment,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        processes.append(worker)
        await wait_ready(
            args.receipts / "worker/NATIVE_READY.json", processes, args.startup_timeout, stopping
        )
        record_cpu_affinity(affinity, "worker", worker)
        log = (args.receipts / "bridge.log").open("wb")
        logs.append(log)
        bridge = await asyncio.create_subprocess_exec(
            *bridge_command,
            env=environment,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        processes.append(bridge)
        await wait_ready(args.listen, processes, args.startup_timeout, stopping)
        record_cpu_affinity(affinity, "bridge", bridge)
        if args.http_port is not None:
            frontend_command = [
                args.frontend_binary,
                "serve",
                "--tts-backend",
                "nemotron",
                "--native-socket",
                str(args.listen),
                "--port",
                str(args.http_port),
                "--host",
                args.http_host,
                "--no-warmup",
            ]
            if args.http_threads is not None:
                frontend_command.extend(["--http-threads", str(args.http_threads)])
            frontend_command = cpu_command(frontend_command, "frontend", taskset, affinity)
            log = (args.receipts / "frontend.log").open("wb")
            logs.append(log)
            frontend = await asyncio.create_subprocess_exec(
                *frontend_command,
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            processes.append(frontend)
            await wait_http(
                args.http_host, args.http_port, processes, args.startup_timeout, stopping
            )
            record_cpu_affinity(affinity, "frontend", frontend)
            report["frontend"] = {
                "command": frontend_command,
                "pid": frontend.pid,
                "host": args.http_host,
                "port": args.http_port,
            }
        report.update(status="READY", socket=str(args.listen))
        (args.receipts / "LAUNCH.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        stop_task = asyncio.create_task(stopping.wait())
        tasks = [stop_task, *(asyncio.create_task(process.wait()) for process in processes)]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if stop_task not in done:
            raise RuntimeError("a supervised process exited unexpectedly")
        report["status"] = "STOPPED"
    except InterruptedError:
        report["status"] = "STOPPED"
    except BaseException as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for process in reversed(processes):
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), args.shutdown_timeout)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await asyncio.wait_for(process.wait(), 5)
        for log in logs:
            log.close()
        report["exit_codes"] = [process.returncode for process in processes]
        (args.receipts / "EXIT.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
