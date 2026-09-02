#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

OUTPUT_RATE = 24000
OUTPUT_PACKET_SAMPLES = 1920


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the VoiceChat realtime protocol")
    parser.add_argument("--server", default="ws://127.0.0.1:8080")
    parser.add_argument("--audio", required=True, help="mono PCM16 WAV input")
    parser.add_argument("--output", help="write received 24 kHz PCM16 to this WAV")
    parser.add_argument("--api-key")
    parser.add_argument("--instructions")
    parser.add_argument("--tools", help="JSON tool array or path to a JSON file")
    parser.add_argument("--tool-result", help="JSON result returned for every tool call")
    parser.add_argument("--require-tool-call", action="store_true")
    parser.add_argument("--tail-seconds", type=float, default=4.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--skip-alias-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def websocket_url(server: str, path: str, api_key: str | None) -> str:
    parsed = urlsplit(server)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    require(scheme in ("ws", "wss") and bool(parsed.netloc), f"invalid server URL: {server}")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if api_key:
        query["api_key"] = api_key
    return urlunsplit((scheme, parsed.netloc, path, urlencode(query), ""))


def load_json_argument(value: str | None):
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = json.loads(Path(value).read_text(encoding="utf-8"))
    require(isinstance(parsed, list), "--tools must contain a JSON array")
    return parsed


def load_pcm(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as source:
        require(source.getnchannels() == 1, "input WAV must be mono")
        require(source.getsampwidth() == 2, "input WAV must contain PCM16")
        rate = source.getframerate()
        require(16000 <= rate <= 48000, "input WAV rate must be between 16 and 48 kHz")
        return rate, source.readframes(source.getnframes())


async def receive_json(socket, timeout: float) -> dict:
    message = await asyncio.wait_for(socket.recv(), timeout=timeout)
    require(isinstance(message, str), "server sent an unexpected binary event")
    event = json.loads(message)
    require(isinstance(event, dict), "server event must be a JSON object")
    return event


async def check_alias(args: argparse.Namespace) -> None:
    import websockets

    url = websocket_url(args.server, "/realtime", args.api_key)
    async with websockets.connect(url, ping_interval=None, open_timeout=args.timeout) as socket:
        created = await receive_json(socket, args.timeout)
        require(created.get("type") == "session.created", "/realtime did not create a session")
        socket_event_id = created.get("event_id")
        require(bool(socket_event_id), "session.created omitted event_id")
        await socket.send("{")
        error = await receive_json(socket, args.timeout)
        require(error.get("type") == "error", "malformed JSON did not produce an error")
        await socket.send(json.dumps({"type": "session.close"}))
        while True:
            event = await receive_json(socket, args.timeout)
            if event.get("type") == "session.end":
                break


async def run_session(args: argparse.Namespace, rate: int, pcm: bytes, tools: list) -> dict:
    import websockets

    url = websocket_url(args.server, "/v1/realtime", args.api_key)
    events: list[dict] = []
    output = bytearray()
    output_times: list[float] = []
    tool_calls = 0
    tool_names: list[str] = []
    tool_results_sent = 0
    packets_after_tool_result = 0
    agent_deltas: list[str] = []
    agent_transcripts: list[str] = []
    user_deltas: list[str] = []
    user_transcripts: list[str] = []

    async with websockets.connect(url, ping_interval=None, open_timeout=args.timeout) as socket:
        created = await receive_json(socket, args.timeout)
        require(created.get("type") == "session.created", "server did not create a session")
        require(
            created.get("session", {})
            .get("audio", {})
            .get("output", {})
            .get("format", {})
            .get("rate")
            == OUTPUT_RATE,
            "session.created advertised the wrong output rate",
        )
        events.append(created)

        session = {
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": rate}},
                "output": {"format": "pcm16"},
            },
            "tools": tools,
        }
        if args.instructions is not None:
            session["instructions"] = args.instructions
        await socket.send(json.dumps({"type": "session.update", "session": session}))

        async def receive_events() -> dict:
            nonlocal tool_calls, tool_results_sent, packets_after_tool_result
            final_stats = None
            while True:
                event = await receive_json(socket, args.timeout)
                events.append(event)
                require(bool(event.get("event_id")), f"{event.get('type')} omitted event_id")
                event_type = event.get("type")
                if event_type == "error":
                    raise RuntimeError(
                        f"server error: {event.get('error', {}).get('message', event)}"
                    )
                if event_type == "session.updated":
                    require(
                        isinstance(event.get("session", {}).get("tools"), list),
                        "session.updated tools must be a JSON array",
                    )
                elif event_type == "response.output_audio.delta":
                    packet = base64.b64decode(event.get("delta", ""), validate=True)
                    require(
                        len(packet) == OUTPUT_PACKET_SAMPLES * 2,
                        "output audio packet is not 80 ms of PCM16",
                    )
                    output.extend(packet)
                    output_times.append(time.monotonic())
                    if tool_results_sent:
                        packets_after_tool_result += 1
                elif event_type == "response.output_audio_transcript.delta":
                    agent_deltas.append(event.get("delta", ""))
                elif event_type == "response.output_audio_transcript.done":
                    agent_transcripts.append(event.get("transcript", ""))
                elif event_type == "conversation.item.input_audio_transcription.delta":
                    user_deltas.append(event.get("delta", ""))
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    user_transcripts.append(event.get("transcript", ""))
                elif event_type == "response.function_call_arguments.done":
                    tool_calls += 1
                    tool_names.append(event.get("name", ""))
                    json.loads(event.get("arguments", "{}"))
                    if args.tool_result is not None:
                        json.loads(args.tool_result)
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "status": "completed",
                                        "call_id": event.get("call_id", ""),
                                        "name": event.get("name", ""),
                                        "output": args.tool_result,
                                    },
                                }
                            )
                        )
                        tool_results_sent += 1
                elif event_type == "session.end":
                    final_stats = event.get("stats", {})
                    break
            return final_stats or {}

        async def send_audio() -> None:
            samples_per_chunk = rate * 80 // 1000
            bytes_per_chunk = samples_per_chunk * 2
            tail_chunks = max(0, round(args.tail_seconds * 1000 / 80))
            chunks = [
                pcm[offset : offset + bytes_per_chunk]
                for offset in range(0, len(pcm), bytes_per_chunk)
            ]
            chunks.extend([bytes(bytes_per_chunk)] * tail_chunks)
            for chunk in chunks:
                if len(chunk) < bytes_per_chunk:
                    chunk += bytes(bytes_per_chunk - len(chunk))
                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                if not args.no_realtime:
                    await asyncio.sleep(0.08)
            await socket.send(json.dumps({"type": "session.close"}))

        receiver = asyncio.create_task(receive_events())
        sender = asyncio.create_task(send_audio())
        try:
            stats = await receiver
            await sender
        except BaseException:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            raise

    event_types = {event.get("type") for event in events}
    required_events = {
        "session.created",
        "session.updated",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
        "session.end",
    }
    require(
        required_events <= event_types, f"missing events: {sorted(required_events - event_types)}"
    )
    require(output, "server returned no audio")
    require(stats.get("chunks_received", 0) > 0, "session.end reported no input")
    require(stats.get("chunks_dropped", -1) == 0, "server dropped input chunks")
    if args.require_tool_call:
        require(tool_calls > 0, "model emitted no tool call")
        if args.tool_result is not None:
            require(tool_results_sent > 0, "client did not return a tool result")
            require(packets_after_tool_result > 0, "server returned no audio after the tool result")

    mean_packet_interval_ms = 0.0
    if len(output_times) > 1:
        intervals = [
            (current - previous) * 1000 for previous, current in zip(output_times, output_times[1:])
        ]
        mean_packet_interval_ms = sum(intervals) / len(intervals)
    return {
        "audio": bytes(output),
        "events": len(events),
        "packets": len(output_times),
        "mean_packet_interval_ms": mean_packet_interval_ms,
        "tool_calls": tool_calls,
        "tool_names": tool_names,
        "tool_results_sent": tool_results_sent,
        "packets_after_tool_result": packets_after_tool_result,
        "assistant_text": " ".join(filter(None, agent_transcripts)) or "".join(agent_deltas),
        "user_text": " ".join(filter(None, user_transcripts)) or "".join(user_deltas),
        "stats": stats,
    }


async def async_main(args: argparse.Namespace) -> dict:
    rate, pcm = load_pcm(Path(args.audio))
    tools = load_json_argument(args.tools)
    if not args.skip_alias_check:
        await check_alias(args)
    return await run_session(args, rate, pcm, tools)


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as sink:
            sink.setnchannels(1)
            sink.setsampwidth(2)
            sink.setframerate(OUTPUT_RATE)
            sink.writeframes(result.pop("audio"))
    else:
        result.pop("audio")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
