from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
from pathlib import Path


def verify_model(root, seal, identity):
    if seal["model_identity"] != identity:
        raise RuntimeError("Deployment identity mismatch")
    for name, row in seal["files"].items():
        p = (root / name).resolve(strict=True)
        if not p.is_relative_to(root.resolve(strict=True)):
            raise RuntimeError("Model path escapes root")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != row["sha256"] or p.stat().st_size != row["bytes"]:
            raise RuntimeError("Model source mismatch: " + name)


def exercise_inflight_lifecycle(engine):
    if any(slot is not None for slot in engine.slots):
        raise RuntimeError("In-flight lifecycle checks require an empty owner")
    text = "Hello from the native three megakernel runner."
    request = engine.engine._requests([text] * engine.batch, None, "english_sample")[0]

    def collect(slot, cancel):
        epoch = engine.epochs[slot] + 1
        identity = f"native-lifecycle-{slot}-{epoch}"
        digest = hashlib.sha256()
        chunks = []
        terminal = None
        try:
            engine.admit(
                slot, epoch, text=text, token_ids=request.target_ids, seed=0, request_id=identity
            )
            for _ in range(engine.steps + 16):
                for event in engine.advance([slot]):
                    if (
                        event["slot"] != slot
                        or event["epoch"] != epoch
                        or event["request_id"] != identity
                    ):
                        raise RuntimeError("Lifecycle execution changed request ownership")
                    if event["type"] == "chunk":
                        digest.update(event["pcm_f32_le"])
                        chunks.append(event["samples"])
                    elif event["type"] == "finish":
                        terminal = event
                if cancel and chunks:
                    terminal = engine.cancel(slot, epoch)
                    break
                if terminal is not None:
                    break
            if terminal is None or not chunks:
                raise RuntimeError("Lifecycle request did not complete or produce audio")
            return {
                "batch": engine.batch,
                "mode": "request-local-inflight",
                "slot": slot,
                "epoch": epoch,
                "cancel_after_first_chunk": cancel,
                "pcm_sha256": digest.hexdigest(),
                "chunk_samples": chunks,
                "finish_reason": terminal["reason"],
                "generations": terminal["receipt"]["generations"],
            }
        finally:
            owner = engine.slots[slot]
            if owner is not None and owner.epoch == epoch and not engine.poisoned:
                if not owner.finished:
                    engine.cancel(slot, epoch)
                engine.retire(slot, epoch)

    baseline = collect(0, False)
    checks = [baseline, collect(0, True), collect(0, False), collect(engine.batch - 1, False)]
    keys = ("pcm_sha256", "chunk_samples", "finish_reason", "generations")
    for check in checks:
        if not check["cancel_after_first_chunk"]:
            if any(check[key] != baseline[key] for key in keys):
                raise RuntimeError("In-flight exact replay failed after cancel or slot reuse")
            check["exact_replay"] = True
        else:
            check["exact_replay"] = None
    if any(slot is not None for slot in engine.slots):
        raise RuntimeError("In-flight lifecycle checks did not release every slot")
    return checks


async def serve(args):
    from nemotron_native_backend.execution.program import (
        load_program,
        load_serving_plan,
        prepare_for_qualification,
    )
    from nemotron_native_backend.execution.readiness import make_readiness_binding

    family = json.loads(args.plan.read_text())["family"]
    neural = load_program(args.neural_plan, family)
    plan = load_serving_plan(args.plan, neural, family)
    identity = plan.contract["model"]["checkpoint_sha256"]
    verify_model(args.model, json.loads(args.model_seal.read_text()), identity)
    pins = json.loads(args.source_seal.read_text())["package_source_pins"]
    readiness = make_readiness_binding(
        args.readiness,
        args.readiness_artifacts,
        args.model,
        source_pins=pins,
        wide_plan=plan,
        model_identity=identity,
        qualification_only=True,
    )
    args.receipts.mkdir(parents=True, exist_ok=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    worker = await prepare_for_qualification(
        plan,
        neural,
        args.neural_plan,
        args.artifacts,
        args.model,
        source_pins=pins,
        model_identity=identity,
        readiness=readiness,
        socket_path=args.socket,
        receipts=args.receipts,
        qualification_only=True,
    )
    try:
        if args.exercise_lifecycle:
            keys = (
                "batch",
                "executed_steps",
                "first_eos_generations",
                "samples",
                "finish_reasons",
                "pcm_sha256",
                "active_codes_sha256",
                "active_phonemes_sha256",
                "live_codes_sha256",
            )
            report = {"status": "RUNNING", "checks": []}
            loop = asyncio.get_running_loop()

            def collect(engine, cancel):
                it = engine.stream(
                    ["Hello from the native three megakernel runner."] * engine.batch,
                    seed=0,
                    epoch=1,
                )
                try:
                    for event in it:
                        if cancel and event["type"] == "chunk":
                            break
                finally:
                    it.close()

            inflight = getattr(worker.backend, "inflight_engine", None)
            for width, engine in worker.backend.engines.items():
                if inflight is not None and engine is inflight.engine:
                    checks = await loop.run_in_executor(
                        worker.backend.executor, exercise_inflight_lifecycle, inflight
                    )
                    report["checks"].extend(checks)
                    worker.batcher._epochs = list(inflight.epochs)
                    continue
                expected = {k: engine.last_receipt[k] for k in keys}
                for cancel in (False, True, False):
                    await loop.run_in_executor(
                        worker.backend.executor, lambda e=engine, c=cancel: collect(e, c)
                    )
                    if not cancel and {k: engine.last_receipt[k] for k in keys} != expected:
                        raise RuntimeError(
                            "Exact replay failed after construction/cancel atB" + str(width)
                        )
                    report["checks"].append(
                        {
                            "batch": width,
                            "cancel_after_first_chunk": cancel,
                            "exact_replay": None if cancel else True,
                        }
                    )
            report["status"] = "PASS_NATIVE_FACTORY_LIFECYCLE"
            (args.receipts / "LIFECYCLE.json").write_text(json.dumps(report, indent=2) + "\n")
        stopping = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopping.set)
        await worker.server.start()
        receipt = dict(
            status="QUALIFICATION_SERVER_LISTENING",
            physical_widths=sorted(worker.backend.engines),
            neural_plan_sha256=neural.manifest_sha256,
            serving_plan_sha256=plan.manifest_sha256,
            production_enabled=False,
        )
        (args.receipts / "NATIVE_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        await stopping.wait()
    finally:
        await worker.server.shutdown()
        await worker.backend.shutdown()


def main():
    p = argparse.ArgumentParser()
    for key in (
        "neural-plan",
        "plan",
        "readiness",
        "readiness-artifacts",
        "artifacts",
        "model",
        "model-seal",
        "source-seal",
        "receipts",
        "socket",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--qualification-only", action="store_true")
    p.add_argument("--exercise-lifecycle", action="store_true")
    args = p.parse_args()
    if not args.qualification_only:
        p.error("--qualification-only is required")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
