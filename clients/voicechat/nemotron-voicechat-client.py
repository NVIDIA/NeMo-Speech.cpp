#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron Voicechat Client
Streams audio from microphone or file to Nemotron VoiceChat server
and plays received audio on the output device.

Supports launching multiple concurrent streams for load testing via
``--num-streams N``.
"""

import argparse
import asyncio
import base64
import datetime
import json
import os
import ssl
import sys
import time
import uuid
from queue import Queue
from threading import Thread

import numpy as np
import soundfile as sf
import websockets

# Audio configuration
INPUT_SAMPLE_RATE = 24000  # 24kHz input sample rate
OUTPUT_SAMPLE_RATE = 24000  # 24kHz output sample rate
CHANNELS = 1
INPUT_CHUNK_DURATION_MS = 80  # 80ms chunks for capture
FILE_INPUT_TRAILING_SILENCE_SECONDS = 20
OUTPUT_CHUNK_DURATION_MS = 160  # 160ms chunks for playback
INPUT_CHUNK_SIZE = int(INPUT_SAMPLE_RATE * INPUT_CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz
OUTPUT_CHUNK_SIZE = int(
    OUTPUT_SAMPLE_RATE * OUTPUT_CHUNK_DURATION_MS / 1000
)  # 3840 samples at 24kHz


# ---------------------------------------------------------------------------
# Dummy tool implementations
# ---------------------------------------------------------------------------


def _tool_convert_currency(amount, from_currency, to_currency, **_):
    """Return a fake currency conversion result."""
    # Dummy exchange rates relative to USD
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50,
        "CAD": 1.36,
        "AUD": 1.53,
        "CHF": 0.89,
        "INR": 83.12,
        "CNY": 7.24,
        "MXN": 17.15,
    }
    from_rate = rates.get(from_currency.upper(), 1.0)
    to_rate = rates.get(to_currency.upper(), 1.0)
    converted = round(float(amount) / from_rate * to_rate, 2)
    return {
        "converted_amount": converted,
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "exchange_rate": round(to_rate / from_rate, 6),
    }


def _tool_get_news_headlines(country=None, category=None, **_):
    """Return dummy news headlines."""
    headlines = [
        "Global markets rally as inflation data comes in lower than expected",
        "Scientists announce breakthrough in renewable energy storage",
        "World leaders gather for climate summit in Geneva",
        "Tech giant unveils next-generation AI assistant",
        "Major earthquake strikes Pacific region; tsunami warnings issued",
    ]
    result = {"headlines": headlines[:3]}
    if country:
        result["country"] = country
    if category:
        result["category"] = category
    result["note"] = "dummy result – not live news"
    return result


def _tool_calculate_bmi(weight, height, **_):
    """Calculate BMI and return the value with category."""
    """ Weight in kg and Height in meter """
    bmi = round(float(weight) / (float(height) ** 2), 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": bmi, "category": category, "weight": weight, "height": height}


def _tool_get_current_datetime(timezone=None, **_):
    """Return the current date and time."""
    now = datetime.datetime.now()
    result = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }
    if timezone:
        result["timezone"] = timezone
        result["note"] = "timezone parameter ignored – local system time returned"
    return result


def _tool_get_current_time(timezone=None, **_):
    """Return the current time."""
    now = datetime.datetime.now()
    result = {"time": now.strftime("%H:%M:%S")}
    if timezone:
        result["timezone"] = timezone
        result["note"] = "timezone parameter ignored – local system time returned"
    return result


TOOL_REGISTRY = {
    "calculate_bmi": _tool_calculate_bmi,
    "convert_currency": _tool_convert_currency,
    "get_current_datetime": _tool_get_current_datetime,
    "get_current_time": _tool_get_current_time,
    "get_news_headlines": _tool_get_news_headlines,
}


def _validate_tools(tools):
    """Validate tools against the OpenAI Realtime API tool spec."""
    if not isinstance(tools, list):
        raise ValueError("--tools must be a JSON array of tool objects")
    for i, tool in enumerate(tools):
        prefix = f"tools[{i}]"
        if not isinstance(tool, dict):
            raise ValueError(f"{prefix}: each tool must be an object")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ValueError(f'{prefix}: "name" must be a non-empty string')
        if "description" in tool and not isinstance(tool["description"], str):
            raise ValueError(f'{prefix}: "description" must be a string')
        if "ack_messages" in tool:
            ack_messages = tool["ack_messages"]
            if not isinstance(ack_messages, list) or not ack_messages:
                raise ValueError(f'{prefix}: "ack_messages" must be a non-empty array of strings')
            if not all(isinstance(message, str) and message for message in ack_messages):
                raise ValueError(f'{prefix}: "ack_messages" must contain only non-empty strings')
        if "parameters" in tool:
            params = tool["parameters"]
            if not isinstance(params, dict):
                raise ValueError(f'{prefix}: "parameters" must be an object')
            if params.get("type") not in ("object", "dict"):
                raise ValueError(f'{prefix}: parameters "type" must be "object"')
            if "properties" in params and not isinstance(params["properties"], dict):
                raise ValueError(f'{prefix}: parameters "properties" must be an object')


def dispatch_tool(name: str, arguments: dict) -> dict:
    """Call the matching dummy tool and return its result, or an error dict."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"result": f"Tool {name} is not available"}
    try:
        return fn(**arguments)
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------


class NemotronVoicechatClient:
    def __init__(
        self,
        uri,
        verbose=False,
        audio_output=None,
        user_text_output=None,
        agent_text_output=None,
        function_text_output=None,
        conversation_output=None,
        input_file=None,
        enable_playback=True,
        max_duration=120,
        api_key=None,
        function_id=None,
        audio_format="pcm16",
        instructions=None,
        tools=None,
        stream_id=None,
        output_dir=None,
        ignore_tool_calls=False,
        trailing_silence=0.0,
    ):
        if not (isinstance(trailing_silence, (int, float)) and 0 <= trailing_silence <= 300):
            raise ValueError(
                f"trailing_silence must be between 0 and 300 seconds, got {trailing_silence!r}"
            )
        self.uri = uri
        self.api_key = api_key
        self.function_id = function_id
        self.stream_id = stream_id
        self.output_dir = output_dir

        # Only import and initialize PyAudio if needed (microphone or playback)
        self.audio = None
        if not input_file or enable_playback:
            try:
                import pyaudio

                self.audio = pyaudio.PyAudio()
            except ImportError:
                if not input_file:
                    raise ImportError(
                        "PyAudio is required for microphone input. "
                        "Install it with: pip install pyaudio\n"
                        "On Ubuntu/Debian: sudo apt-get install portaudio19-dev && pip install pyaudio"
                    )
                elif enable_playback:
                    raise ImportError(
                        "PyAudio is required for audio playback. "
                        "Install it with: pip install pyaudio, or use --no-playback flag\n"
                        "On Ubuntu/Debian: sudo apt-get install portaudio19-dev && pip install pyaudio"
                    )

        self.running = False
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = Queue(maxsize=50)  # Buffer up to 50 chunks (~4 seconds)
        self.verbose = verbose
        self.ignore_tool_calls = ignore_tool_calls
        self.chunks_sent = 0
        self.chunks_received = 0
        self.websocket = None
        self.min_buffer_chunks = 5  # Wait for 5 chunks (~400ms) before starting playback
        self.enable_playback = enable_playback
        self.max_duration = (
            max_duration  # Maximum session duration in seconds (only for microphone)
        )

        # Output file paths
        self.audio_output = audio_output
        self.user_text_output = user_text_output
        self.agent_text_output = agent_text_output
        self.function_text_output = function_text_output
        self.conversation_output = conversation_output

        # Input file for streaming
        self.input_file = input_file
        self.trailing_silence = trailing_silence
        self.input_audio_data = None
        self.input_audio_index = 0
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.input_chunk_size = INPUT_CHUNK_SIZE
        self.instructions = instructions
        self.tools = tools

        # Audio format configuration (pcm16 only)
        self.audio_format = audio_format

        # Buffer for saving received audio to file
        self.received_audio_buffer = []

        # Buffer for saving input microphone audio
        self.input_audio_buffer = []

        # Buffers for accumulating delta text fragments
        self.current_user_text = []
        self.current_agent_text = []

        # Completed tool calls accumulated for file output
        self.completed_function_calls = []

        # Track completed transcripts
        self.completed_user_transcripts = []
        self.completed_agent_transcripts = []

        # Conversation log for structured output
        self.conversation_log = []

        # Timing statistics
        self.start_time = None
        self.first_response_time = None
        self.last_chunk_time = None
        self.inter_chunk_latencies = []

        # Error tracking (for multi-stream summary)
        self._error = None

        # Server-side stats (populated from session.end message)
        self.server_stats = None
        self._session_end_event = asyncio.Event()

    def _log(self, msg):
        """Print with an optional ``[stream N]`` prefix."""
        if self.stream_id is not None:
            print(f"[stream {self.stream_id}] {msg}")
        else:
            print(msg)

    def _output_path(self, filename):
        """Prepend output_dir and stream_id prefix when set."""
        if self.stream_id is not None:
            filename = f"stream{self.stream_id}_{filename}"
        if self.output_dir:
            filename = os.path.join(self.output_dir, filename)
        return filename

    def load_input_file(self):
        """Load file audio, preserving sample rates accepted by the server."""
        if not self.input_file:
            return

        try:
            # Read audio file
            audio_data, sample_rate = sf.read(self.input_file, dtype='float32')

            # Convert stereo to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Pass the file's sample rate through to the server — it accepts
            # any rate in [16000, 48000] and resamples to 16kHz internally.
            # Files outside that range are rejected by the server with a clear
            # error; no client-side resampling is needed.
            if 16000 <= sample_rate <= 48000:
                self.input_sample_rate = sample_rate
                self.input_chunk_size = int(sample_rate * INPUT_CHUNK_DURATION_MS / 1000)
            else:
                raise ValueError(
                    f"Unsupported sample rate {sample_rate}Hz in {self.input_file}. Server accepts 16000-48000 Hz."
                )

            if self.trailing_silence > 0:
                silence_samples = int(self.trailing_silence * self.input_sample_rate)
                audio_data = np.concatenate(
                    [audio_data, np.zeros(silence_samples, dtype=np.float32)]
                )
                self._log(f"Added {self.trailing_silence:.1f}s silence ({silence_samples} samples)")

            self.input_audio_data = audio_data
            duration = len(audio_data) / self.input_sample_rate
            self._log(
                f"Loaded input file: {self.input_file} "
                f"({duration:.2f}s, {self.input_sample_rate}Hz, mono)"
            )

        except Exception as e:
            self._log(f"Error loading input file: {e}")
            raise

    def setup_audio_streams(self):
        """Initialize audio input and output streams"""
        # Only proceed if audio streams are actually needed
        if not self.audio:
            return

        import pyaudio  # Import here since it's only needed if audio streams are used

        # Input stream: only open if not using file input
        if not self.input_file:
            self.input_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=INPUT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=INPUT_CHUNK_SIZE,
            )

        # Output stream: only open if playback is enabled
        if self.enable_playback:
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=OUTPUT_SAMPLE_RATE,
                output=True,
                frames_per_buffer=OUTPUT_CHUNK_SIZE,
            )

        if self.verbose:
            self._log(
                f"Audio streams initialized: Input {INPUT_SAMPLE_RATE}Hz ({INPUT_CHUNK_SIZE} samples), Output {OUTPUT_SAMPLE_RATE}Hz ({OUTPUT_CHUNK_SIZE} samples), {CHANNELS} channel(s)"
            )

    async def _request_session_end(self, websocket, timeout=5.0):
        """Send session.close and wait for the server's session.end response."""
        try:
            close_msg = {"type": "session.close", "event_id": str(uuid.uuid4())}
            await websocket.send(json.dumps(close_msg))
        except Exception:
            return
        try:
            await asyncio.wait_for(self._session_end_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            if self.verbose:
                self._log("Timed out waiting for session.end from server")

    async def send_audio(self, websocket):
        """Capture audio from mic or file and send to WebSocket as JSON"""
        try:
            # Get event loop for running blocking I/O (only needed for mic)
            loop = asyncio.get_event_loop() if not self.input_file else None

            # Track session start time (only for microphone duration limit)
            session_start_time = time.time() if not self.input_file else None

            prev_chunk_time = None

            while self.running:
                # Get audio data from file or microphone
                if self.input_file:

                    # Read from file
                    if self.input_audio_index >= len(self.input_audio_data):
                        self._log("End of input file reached, waiting for remaining responses...")
                        await asyncio.sleep(2.0)
                        await self._request_session_end(websocket)
                        self.running = False
                        try:
                            await websocket.close()
                        except Exception:
                            pass
                        break

                    # Extract one chunk
                    chunk_end = min(
                        self.input_audio_index + self.input_chunk_size, len(self.input_audio_data)
                    )
                    audio_float32 = self.input_audio_data[self.input_audio_index : chunk_end]
                    self.input_audio_index = chunk_end

                    # Pad with zeros if last chunk is incomplete
                    if len(audio_float32) < self.input_chunk_size:
                        audio_float32 = np.pad(
                            audio_float32, (0, self.input_chunk_size - len(audio_float32))
                        )

                    # For file input mode, sleep only for remaining time to reach 80ms
                    if prev_chunk_time is not None:
                        elapsed_time = time.time() - prev_chunk_time
                        target_duration = INPUT_CHUNK_DURATION_MS / 1000.0  # 80ms
                        remaining_time = target_duration - elapsed_time
                        if remaining_time > 0:
                            await asyncio.sleep(remaining_time)
                    prev_chunk_time = time.time()

                else:
                    # Read from microphone
                    # Check if max duration reached (only for microphone)
                    if self.max_duration and session_start_time:
                        elapsed = time.time() - session_start_time
                        if elapsed >= self.max_duration:
                            self._log(
                                f"Maximum session duration ({self.max_duration}s) reached, ending session..."
                            )
                            await asyncio.sleep(2.0)
                            await self._request_session_end(websocket)
                            self.running = False
                            try:
                                await websocket.close()
                            except Exception:
                                pass
                            break

                    try:
                        audio_data = await loop.run_in_executor(
                            None,
                            self.input_stream.read,
                            INPUT_CHUNK_SIZE,
                            False,  # exception_on_overflow
                        )
                    except OSError:
                        # Stream was stopped, exit gracefully
                        if not self.running:
                            break
                        raise

                    if len(audio_data) == 0:
                        if self.verbose:
                            self._log("Warning: Empty audio data from microphone")
                        continue

                    # Convert PCM16 bytes to numpy float32 array normalized to [-1.0, 1.0]
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Save input audio for later export
                    self.input_audio_buffer.append(audio_float32)

                # Convert float32 to int16 PCM
                audio_int16 = (audio_float32 * 32768.0).astype(np.int16)
                encoded_data = audio_int16.tobytes()

                # Log first chunk
                if self.verbose and self.chunks_sent == 0:
                    self._log(f"First chunk: PCM16 - {len(encoded_data)} bytes")

                # Encode data as base64
                audio_base64 = base64.b64encode(encoded_data).decode('utf-8')

                # Create JSON message
                event_id = str(uuid.uuid4())
                message = {
                    "type": "input_audio_buffer.append",
                    "event_id": event_id,
                    "audio": audio_base64,
                }

                # Send JSON to WebSocket
                await websocket.send(json.dumps(message))
                self.chunks_sent += 1

                if self.verbose and self.chunks_sent % 50 == 0 and self.chunks_sent > 0:
                    self._log(f"Sent {self.chunks_sent} audio chunks")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.running:
                self._log(f"Error sending audio: {e}")
                import traceback

                traceback.print_exc()

    async def receive_audio(self, websocket):
        """Receive audio from WebSocket and play on output device"""
        try:
            while self.running:
                # Receive message from WebSocket
                message = await websocket.recv()

                # Parse as JSON
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")

                    if msg_type == "session.created":
                        # Handle session created message
                        if self.verbose:
                            session_info = data.get("session", {})
                            self._log(f"Session created: {session_info.get('id', 'unknown')}")
                            self._log(
                                f"  Input: {session_info.get('audio', {}).get('input', {}).get('format', {})}"
                            )
                            self._log(
                                f"  Output: {session_info.get('audio', {}).get('output', {}).get('format', {})}"
                            )

                    elif msg_type == "response.output_audio.delta":
                        # Handle audio chunk
                        if "delta" in data:
                            # Decode base64 to get audio data
                            audio_data = base64.b64decode(data["delta"])

                            try:
                                # Convert PCM16 bytes to float32
                                pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
                                pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

                                if pcm_float32.shape[-1] > 0:  # Only queue if we got data
                                    # Save to buffer for WAV file
                                    self.received_audio_buffer.append(pcm_float32)

                                    # Convert float32 [-1.0, 1.0] to int16 for playback
                                    pcm_int16 = (pcm_float32 * 32768.0).astype(np.int16)
                                    pcm_bytes = pcm_int16.tobytes()

                                    # Queue for playback only if enabled
                                    if self.enable_playback:
                                        self.audio_queue.put(pcm_bytes)

                                    self.chunks_received += 1

                                    # Track timing statistics
                                    current_time = time.time()
                                    if self.chunks_received == 1:
                                        # First response timing
                                        self.first_response_time = current_time
                                    elif self.last_chunk_time is not None:
                                        # Inter-chunk latency
                                        latency_ms = (current_time - self.last_chunk_time) * 1000
                                        self.inter_chunk_latencies.append(latency_ms)
                                    self.last_chunk_time = current_time

                                    if self.verbose:
                                        if self.chunks_received == 1:
                                            self._log(
                                                f"Received first audio chunk - PCM16 {len(audio_data)} bytes -> PCM16 {len(pcm_bytes)} bytes ({pcm_int16.shape[0]} samples) (queue: {self.audio_queue.qsize()})"
                                            )
                                        elif self.chunks_received % 50 == 0:
                                            self._log(
                                                f"Received {self.chunks_received} audio chunks (queue: {self.audio_queue.qsize()})"
                                            )
                            except Exception as e:
                                if self.verbose:
                                    self._log(f"Error decoding audio: {e}")
                                    import traceback

                                    traceback.print_exc()
                                continue

                    elif msg_type == "response.output_audio_transcript.delta":
                        # Handle agent text response - accumulate deltas
                        text = data.get("delta", "")
                        if text:
                            self._log(f"[{time.time() - self.start_time:.1f}s][Agent]: {text}")
                            self.current_agent_text.append(text)

                    elif msg_type == "conversation.item.input_audio_transcription.delta":
                        # Handle ASR (speech recognition) text - accumulate deltas
                        asr_text = data.get("delta", "")
                        if asr_text:
                            self._log(f"[{time.time() - self.start_time:.1f}s][You]: {asr_text}")
                            self.current_user_text.append(asr_text)

                    elif msg_type == "conversation.item.input_audio_transcription.completed":
                        # Handle completed user transcript
                        transcript = data.get("transcript", "")
                        if transcript:
                            # Add to completed transcripts
                            self.completed_user_transcripts.append(transcript)
                            # Clear current delta buffer
                            self.current_user_text = []

                            # Display only the completed segment
                            self._log(f"\n[User Transcript Complete]: {transcript}\n")

                            # Add to conversation log
                            self.conversation_log.append(
                                {
                                    "role": "user",
                                    "content": transcript,
                                    "timestamp": time.time() - self.start_time,
                                }
                            )

                    elif msg_type == "response.output_audio_transcript.done":
                        # Handle completed agent text output
                        transcript = data.get("transcript", "")
                        if transcript:
                            # Add to completed transcripts
                            self.completed_agent_transcripts.append(transcript)
                            # Clear current delta buffer
                            self.current_agent_text = []

                            # Display only the completed segment
                            self._log(f"\n[Agent Text Complete]: {transcript}\n")

                            # Add to conversation log
                            self.conversation_log.append(
                                {
                                    "role": "agent",
                                    "content": transcript,
                                    "timestamp": time.time() - self.start_time,
                                }
                            )

                    elif msg_type == "response.function_call_arguments.done":
                        # Full tool call received — arguments are complete
                        call_id = data.get("call_id", "")
                        name = data.get("name", "")
                        arguments = data.get("arguments", "")
                        try:
                            parsed_args = json.loads(arguments) if arguments else {}
                        except json.JSONDecodeError:
                            parsed_args = arguments
                        print(f"\n[Tool Call]: {name}({json.dumps(parsed_args)})")

                        if self.ignore_tool_calls:
                            print("[Tool Call]: ignored (--ignore-tool-calls)\n")
                            result = None
                        else:
                            result = json.dumps(dispatch_tool(name, parsed_args))
                            print(f"[Tool Result]: {result}\n")
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "status": "completed",
                                            "call_id": call_id,
                                            "name": name,
                                            "arguments": json.dumps(parsed_args),
                                            "output": result,
                                        },
                                    }
                                )
                            )
                        # Accumulate for file output
                        self.completed_function_calls.append(
                            {
                                "name": name,
                                "arguments": parsed_args,
                                "result": result,
                                "call_id": call_id,
                                "timestamp": time.time(),
                            }
                        )
                        # Add to conversation log
                        self.conversation_log.append(
                            {
                                "role": "tool_call",
                                "content": {
                                    "name": name,
                                    "arguments": parsed_args,
                                    "result": result,
                                },
                                "call_id": call_id,
                                "timestamp": time.time(),
                            }
                        )

                    elif msg_type == "error":
                        # Handle error messages
                        error_info = data.get("error", {})
                        error_msg = error_info.get("message", "Unknown error")
                        error_code = error_info.get("code", "unknown")
                        self._log(f"[Error] {error_code}: {error_msg}")

                    elif msg_type == "input_audio_buffer.speech_started":
                        if self.verbose:
                            self._log("Speech started")

                    elif msg_type == "input_audio_buffer.speech_stopped":
                        if self.verbose:
                            self._log("Speech stopped")

                    elif msg_type == "session.updated":
                        # Handle session update acknowledgment
                        if self.verbose:
                            self._log("Session updated")

                    elif msg_type == "session.end":
                        stats = data.get("stats", {})
                        self.server_stats = stats
                        self._session_end_event.set()
                        dropped = stats.get("chunks_dropped", 0)
                        if dropped > 0:
                            self._log(
                                f"[Server Stats] chunks_received={stats.get('chunks_received')}, "
                                f"chunks_sent={stats.get('chunks_sent')}, "
                                f"chunks_dropped={dropped}, "
                                f"triton_inferences={stats.get('triton_inferences')}, "
                                f"audio_duration={stats.get('audio_duration_received_s')}s"
                            )
                        elif self.verbose:
                            self._log(f"[Server Stats] {stats}")

                    else:
                        # Other message types
                        if self.verbose:
                            self._log(f"Received message: {msg_type}")

                except json.JSONDecodeError as e:
                    self._log(f"Failed to parse JSON message: {e}")

        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            # Connection closed, exit gracefully
            pass
        except Exception as e:
            if self.running:
                self._log(f"Error receiving audio: {e}")
                import traceback

                traceback.print_exc()

    def play_audio_thread(self):
        """Thread to play audio from queue with buffering"""
        # Wait for initial buffer to fill
        if self.verbose:
            self._log(f"Waiting for {self.min_buffer_chunks} chunks before starting playback...")
        while self.running and self.audio_queue.qsize() < self.min_buffer_chunks:
            time.sleep(0.01)

        if not self.running:
            return

        if self.verbose:
            self._log(f"Starting playback with {self.audio_queue.qsize()} chunks buffered...")

        underrun_count = 0
        consecutive_underruns = 0
        chunks_played = 0
        last_log_time = time.time()

        while self.running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                consecutive_underruns = 0  # Reset on successful read
                try:
                    self.output_stream.write(audio_data)
                    chunks_played += 1

                    # Log every second during playback to show progress
                    if self.verbose and time.time() - last_log_time >= 1.0:
                        self._log(
                            f"Playback: {chunks_played} played, {self.chunks_received} received, queue: {self.audio_queue.qsize()}"
                        )
                        last_log_time = time.time()
                except Exception as e:
                    if self.running:
                        self._log(f"Error playing audio: {e}")
            else:
                # Queue empty - potential buffer underrun
                consecutive_underruns += 1
                # Only count as real underrun if empty longer than one chunk duration (160ms)
                if (
                    consecutive_underruns == 32
                ):  # 32 iterations * 5ms = 160ms gap (one chunk duration)
                    underrun_count += 1
                    if (
                        self.verbose and underrun_count % 10 == 1
                    ):  # Log every 10th real underrun in verbose mode
                        self._log(
                            f"Warning: Playback buffer underrun #{underrun_count} (played: {chunks_played}, received: {self.chunks_received}, queue empty for {consecutive_underruns * 5}ms)"
                        )
                time.sleep(0.005)  # Wait a bit for more data

    async def run(self):
        """Main run loop"""
        self.running = True

        # Load input file if specified
        if self.input_file:
            self.load_input_file()

        self.setup_audio_streams()

        # Start playback thread (only if playback is enabled)
        playback_thread = None
        if self.enable_playback:
            playback_thread = Thread(target=self.play_audio_thread, daemon=True)
            playback_thread.start()

        send_task = None
        receive_task = None

        try:
            if self.verbose:
                self._log(f"Connecting to {self.uri}...")

            # Create SSL context for secure connections
            ssl_context = None
            if self.uri.startswith('wss://'):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            # Add NVCF authentication headers if provided
            additional_headers = {}
            if "nvcf" in self.uri and self.api_key and self.function_id:
                additional_headers = {
                    "authorization": f"Bearer {self.api_key}",
                    "function-id": self.function_id,
                }

            async with websockets.connect(
                self.uri, ssl=ssl_context, additional_headers=additional_headers
            ) as websocket:
                self.websocket = websocket
                self.start_time = time.time()  # Track connection start time
                if self.verbose:
                    self._log("Connected!")

                # Send session.update to configure audio format
                session_update_msg = {
                    "type": "session.update",
                    "event_id": str(uuid.uuid4()),
                    "session": {
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.input_sample_rate,
                                }
                            },
                            "output": {"format": self.audio_format},
                        },
                        "instructions": self.instructions,
                        "tools": self.tools,
                    },
                }
                await websocket.send(json.dumps(session_update_msg))
                if self.verbose:
                    self._log(f"Configured audio format: {self.audio_format}")
                    self._log("Streaming audio...")

                # Run send and receive tasks concurrently
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive_audio(websocket))

                # Wait for both tasks
                await asyncio.gather(send_task, receive_task)

        except websockets.exceptions.WebSocketException as e:
            self._error = str(e)
            self._log(f"WebSocket error: {e}")
        except asyncio.CancelledError:
            self._log("\nCancelled...")
        except Exception as e:
            self._error = str(e)
            self._log(f"Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Set running to False to signal tasks to exit
            self.running = False

            # Cancel tasks if they exist and are still running
            tasks_to_wait = []
            if send_task and not send_task.done():
                send_task.cancel()
                tasks_to_wait.append(send_task)
            if receive_task and not receive_task.done():
                receive_task.cancel()
                tasks_to_wait.append(receive_task)

            # Wait briefly for tasks to complete cancellation
            if tasks_to_wait:
                try:
                    await asyncio.wait(tasks_to_wait, timeout=0.5)
                except Exception:
                    pass

            self.cleanup()

    def get_metrics(self):
        """Return a dict of timing/throughput metrics for this stream."""
        ttfr_ms = None
        if self.start_time and self.first_response_time:
            ttfr_ms = (self.first_response_time - self.start_time) * 1000

        latencies = self.inter_chunk_latencies
        srv = self.server_stats or {}
        return {
            "stream_id": self.stream_id,
            "chunks_sent": self.chunks_sent,
            "chunks_received": self.chunks_received,
            "ttfr_ms": ttfr_ms,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
            "min_latency_ms": float(np.min(latencies)) if latencies else None,
            "max_latency_ms": float(np.max(latencies)) if latencies else None,
            "std_latency_ms": float(np.std(latencies)) if latencies else None,
            "error": self._error,
            "server_chunks_dropped": srv.get("chunks_dropped", 0),
            "server_triton_inferences": srv.get("triton_inferences", 0),
            "server_audio_duration_s": srv.get("audio_duration_received_s", 0),
        }

    def save_received_audio(self, filename="received_audio.wav"):
        """Save all received audio to a WAV file as 16-bit PCM mono"""
        if not self.received_audio_buffer:
            self._log("No audio received to save")
            return

        # Concatenate all received audio chunks (float32, -1.0 to 1.0)
        combined_audio = np.concatenate(self.received_audio_buffer)

        # Convert to int16 PCM
        audio_int16 = (combined_audio * 32768.0).astype(np.int16)

        # Save to WAV file (16-bit PCM mono)
        sf.write(filename, audio_int16, OUTPUT_SAMPLE_RATE, subtype='PCM_16')
        duration = len(combined_audio) / OUTPUT_SAMPLE_RATE
        self._log(
            f"Saved {len(self.received_audio_buffer)} chunks ({duration:.2f}s) to {filename} (16-bit PCM mono)"
        )

    def save_input_audio(self, filename="input_audio.wav"):
        """Save all input microphone audio to a WAV file as 16-bit PCM mono"""
        if not self.input_audio_buffer:
            if self.verbose:
                self._log("No input audio to save")
            return

        # Concatenate all input audio chunks (float32, -1.0 to 1.0)
        combined_audio = np.concatenate(self.input_audio_buffer)

        # Convert to int16 PCM
        audio_int16 = (combined_audio * 32768.0).astype(np.int16)

        # Save to WAV file (16-bit PCM mono)
        sf.write(filename, audio_int16, INPUT_SAMPLE_RATE, subtype='PCM_16')
        duration = len(combined_audio) / INPUT_SAMPLE_RATE
        self._log(
            f"Saved input audio ({len(self.input_audio_buffer)} chunks, {duration:.2f}s) to {filename} (16-bit PCM mono)"
        )

    def save_text_logs(
        self, user_text, agent_text, user_filename="user_text.txt", agent_filename="agent_text.txt"
    ):
        """Save user and agent text to separate files"""
        # Save user text
        if user_text:
            try:
                with open(user_filename, 'w', encoding='utf-8') as f:
                    f.write(user_text)
                self._log(f"Saved user text to {user_filename}")
            except Exception as e:
                self._log(f"Error saving user text: {e}")
        elif self.verbose:
            self._log("No user text to save")

        # Save agent text
        if agent_text:
            try:
                with open(agent_filename, 'w', encoding='utf-8') as f:
                    f.write(agent_text)
                self._log(f"Saved agent text to {agent_filename}")
            except Exception as e:
                self._log(f"Error saving agent text: {e}")
        elif self.verbose:
            self._log("No agent text to save")

    def save_function_call_log(self, filename="function_calls.jsonl"):
        """Save all completed tool calls to a JSONL file (one call per line)."""
        if not self.completed_function_calls:
            if self.verbose:
                print("No function calls to save")
            return
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.completed_function_calls:
                    entry_copy = entry.copy()
                    entry_copy['timestamp'] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(entry['timestamp'])
                    )
                    f.write(json.dumps(entry_copy, ensure_ascii=False) + '\n')
            print(f"Saved {len(self.completed_function_calls)} function call(s) to {filename}")
        except Exception as e:
            print(f"Error saving function call log: {e}")

    def save_conversation_log(self, filename="conversation.jsonl"):
        """Save conversation log in JSONL format (one JSON object per line)"""
        if not self.conversation_log:
            if self.verbose:
                self._log("No conversation to save")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.conversation_log:
                    entry_copy = entry.copy()
                    t = entry['timestamp']
                    entry_copy['timestamp'] = f"{int(t) // 60}:{t % 60:05.2f}"
                    f.write(json.dumps(entry_copy, ensure_ascii=False) + '\n')
            self._log(f"Saved conversation log to {filename}")
        except Exception as e:
            self._log(f"Error saving conversation log: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.verbose:
            self._log("\nCleaning up...")
        self.running = False

        # Stop streams immediately to unblock any pending reads/writes
        if self.input_stream and self.input_stream.is_active():
            try:
                self.input_stream.stop_stream()
            except Exception:
                pass

        if self.output_stream and self.output_stream.is_active():
            try:
                self.output_stream.stop_stream()
            except Exception:
                pass

        # Brief pause for threads to notice
        time.sleep(0.05)

        # Show statistics
        if self.verbose:
            self._log(f"Total chunks sent: {self.chunks_sent}")
            self._log(f"Total chunks received: {self.chunks_received}")

        # Show timing statistics
        self._log("\n=== Timing Statistics ===")
        if self.start_time and self.first_response_time:
            ttfr = (self.first_response_time - self.start_time) * 1000  # Convert to ms
            self._log(f"Time to first response: {ttfr:.1f}ms")
        else:
            self._log("Time to first response: N/A (no response received)")

        if len(self.inter_chunk_latencies) > 0:
            mean_latency = np.mean(self.inter_chunk_latencies)
            min_latency = np.min(self.inter_chunk_latencies)
            max_latency = np.max(self.inter_chunk_latencies)
            std_latency = np.std(self.inter_chunk_latencies)
            self._log(
                f"Inter-chunk latency: mean={mean_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms, std={std_latency:.1f}ms"
            )
            self._log(f"Total chunks tracked: {len(self.inter_chunk_latencies)}")
        else:
            self._log("Inter-chunk latency: N/A (insufficient data)")

        if self.server_stats:
            s = self.server_stats
            self._log(
                f"\n=== Server Stats ===\n"
                f"Input frames received: {s.get('chunks_received', 'N/A')}\n"
                f"Input frames dropped:  {s.get('chunks_dropped', 0)}\n"
                f"Output chunks sent:    {s.get('chunks_sent', 'N/A')}\n"
                f"Triton inferences:     {s.get('triton_inferences', 'N/A')}\n"
                f"Audio duration received: {s.get('audio_duration_received_s', 'N/A')}s"
            )

        # Save received audio to WAV file
        try:
            if self.audio_output:
                audio_filename = self.audio_output
            else:
                audio_filename = self._output_path("received_audio.wav")
            self.save_received_audio(audio_filename)
        except Exception as e:
            self._log(f"Error saving audio: {e}")

        # Save input microphone audio to WAV file (only if using microphone, not file input)
        if not self.input_file:
            try:
                input_filename = self._output_path("input_audio.wav")
                self.save_input_audio(input_filename)
            except Exception as e:
                self._log(f"Error saving input audio: {e}")

        # Save text logs to separate files
        try:
            # Combine completed transcripts with current delta buffer
            user_parts = self.completed_user_transcripts.copy()
            if self.current_user_text:
                user_parts.append("".join(self.current_user_text))
            user_text = " ".join(user_parts) if user_parts else ""

            agent_parts = self.completed_agent_transcripts.copy()
            if self.current_agent_text:
                agent_parts.append("".join(self.current_agent_text))
            agent_text = " ".join(agent_parts) if agent_parts else ""

            user_filename = (
                self.user_text_output
                if self.user_text_output
                else self._output_path("user_text.txt")
            )
            agent_filename = (
                self.agent_text_output
                if self.agent_text_output
                else self._output_path("agent_text.txt")
            )
            self.save_text_logs(user_text, agent_text, user_filename, agent_filename)
        except Exception as e:
            self._log(f"Error saving text logs: {e}")

        # Flush any in-progress delta buffers into conversation log
        if self.current_user_text and self.start_time:
            self.conversation_log.append(
                {
                    "role": "user",
                    "content": "".join(self.current_user_text),
                    "timestamp": time.time() - self.start_time,
                }
            )
        if self.current_agent_text and self.start_time:
            self.conversation_log.append(
                {
                    "role": "agent",
                    "content": "".join(self.current_agent_text),
                    "timestamp": time.time() - self.start_time,
                }
            )

        # Save function call log
        try:
            function_filename = (
                self.function_text_output if self.function_text_output else f"function_calls.jsonl"
            )
            self.save_function_call_log(function_filename)
        except Exception as e:
            print(f"Error saving function call log: {e}")

        # Save conversation log
        try:
            conversation_filename = (
                self.conversation_output
                if self.conversation_output
                else self._output_path("conversation.jsonl")
            )
            self.save_conversation_log(conversation_filename)
        except Exception as e:
            self._log(f"Error saving conversation log: {e}")

        # Close streams
        if self.input_stream:
            try:
                self.input_stream.close()
            except Exception:
                pass

        if self.output_stream:
            try:
                self.output_stream.close()
            except Exception:
                pass

        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass

        if self.verbose:
            self._log("Cleanup complete")


# ========================================================================= #
#  Single-stream entry point                                                 #
# ========================================================================= #


async def main(
    websocket_uri,
    verbose,
    audio_output,
    user_text_output,
    agent_text_output,
    function_text_output,
    conversation_output,
    input_file,
    enable_playback,
    max_duration,
    api_key=None,
    function_id=None,
    audio_format="pcm16",
    instructions=None,
    tools=None,
    ignore_tool_calls=False,
    trailing_silence=0.0,
):
    # Create and run client
    client = NemotronVoicechatClient(
        websocket_uri,
        verbose=verbose,
        audio_output=audio_output,
        user_text_output=user_text_output,
        agent_text_output=agent_text_output,
        function_text_output=function_text_output,
        conversation_output=conversation_output,
        input_file=input_file,
        enable_playback=enable_playback,
        max_duration=max_duration,
        api_key=api_key,
        function_id=function_id,
        audio_format=audio_format,
        instructions=instructions,
        tools=tools,
        ignore_tool_calls=ignore_tool_calls,
        trailing_silence=trailing_silence,
    )
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.cleanup()


# ========================================================================= #
#  Multi-stream entry point                                                 #
# ========================================================================= #


async def run_multi_stream(
    websocket_uri,
    num_streams,
    input_file,
    verbose=False,
    audio_format="pcm16",
    api_key=None,
    function_id=None,
    instructions=None,
    tools=None,
    output_dir=None,
):
    """Launch *num_streams* concurrent sessions streaming the same input file."""

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clients = []
    for i in range(num_streams):
        client = NemotronVoicechatClient(
            websocket_uri,
            verbose=verbose,
            input_file=input_file,
            enable_playback=False,  # no playback in multi-stream mode
            max_duration=300,
            api_key=api_key,
            function_id=function_id,
            audio_format=audio_format,
            instructions=instructions,
            tools=tools,
            stream_id=i,
            output_dir=output_dir,
        )
        clients.append(client)

    print(f"\nLaunching {num_streams} concurrent streams...\n")

    # Run all streams concurrently; isolate per-stream failures
    async def _safe_run(client):
        try:
            await client.run()
        except Exception as e:
            client._error = str(e)
            client._log(f"Stream failed: {e}")

    await asyncio.gather(*[_safe_run(c) for c in clients])

    # ---- Aggregate summary ------------------------------------------------
    TABLE_W = 105
    print("\n" + "=" * TABLE_W)
    print("  MULTI-STREAM SUMMARY")
    print("=" * TABLE_W)

    header = (
        f"{'Stream':>6s}  {'Sent':>6s}  {'Recv':>6s}  {'Drop':>6s}  "
        f"{'TTFR(ms)':>10s}  {'Mean(ms)':>10s}  {'Min(ms)':>9s}  "
        f"{'Max(ms)':>9s}  {'Std(ms)':>9s}  {'Status':>8s}"
    )
    print(header)
    print("-" * TABLE_W)

    all_ttfr = []
    all_mean = []
    all_min = []
    all_max = []
    all_std = []
    total_dropped = 0
    errors = 0

    for c in clients:
        m = c.get_metrics()
        has_error = m["error"] is not None
        no_audio = m["chunks_sent"] > 0 and m["chunks_received"] == 0
        if has_error:
            status = "FAIL"
        elif no_audio:
            status = "NO_AUDIO"
        else:
            status = "OK"
        if has_error or no_audio:
            errors += 1

        dropped = m.get("server_chunks_dropped", 0)
        total_dropped += dropped

        ttfr_str = f"{m['ttfr_ms']:.1f}" if m["ttfr_ms"] is not None else "N/A"
        mean_str = f"{m['mean_latency_ms']:.1f}" if m["mean_latency_ms"] is not None else "N/A"
        min_str = f"{m['min_latency_ms']:.1f}" if m["min_latency_ms"] is not None else "N/A"
        max_str = f"{m['max_latency_ms']:.1f}" if m["max_latency_ms"] is not None else "N/A"
        std_str = f"{m['std_latency_ms']:.1f}" if m["std_latency_ms"] is not None else "N/A"

        print(
            f"{m['stream_id']:>6d}  {m['chunks_sent']:>6d}  {m['chunks_received']:>6d}  {dropped:>6d}  "
            f"{ttfr_str:>10s}  {mean_str:>10s}  {min_str:>9s}  "
            f"{max_str:>9s}  {std_str:>9s}  {status:>8s}"
        )

        if m["ttfr_ms"] is not None:
            all_ttfr.append(m["ttfr_ms"])
        if m["mean_latency_ms"] is not None:
            all_mean.append(m["mean_latency_ms"])
        if m["min_latency_ms"] is not None:
            all_min.append(m["min_latency_ms"])
        if m["max_latency_ms"] is not None:
            all_max.append(m["max_latency_ms"])
        if m["std_latency_ms"] is not None:
            all_std.append(m["std_latency_ms"])

    print("-" * TABLE_W)

    # Aggregate row — averages of per-stream values
    if all_ttfr:
        print(
            f"{'AGG':>6s}  {'':>6s}  {'':>6s}  {total_dropped:>6d}  "
            f"{np.mean(all_ttfr):>10.1f}  {np.mean(all_mean):>10.1f}  "
            f"{np.mean(all_min):>9.1f}  {np.mean(all_max):>9.1f}  "
            f"{np.mean(all_std):>9.1f}  "
            f"{f'{len(all_ttfr)}/{num_streams} ok':>8s}"
        )
    print(f"\nStreams: {num_streams}  Succeeded: {num_streams - errors}  Failed: {errors}")

    # Warn if streams with the same input received very different output counts
    metrics = [c.get_metrics() for c in clients]
    sent_groups = {}
    for m in metrics:
        sent_groups.setdefault(m["chunks_sent"], []).append(m)
    for sent, group in sent_groups.items():
        if len(group) < 2:
            continue
        recvs = [g["chunks_received"] for g in group]
        lo, hi = min(recvs), max(recvs)
        if hi > 0 and lo < hi * 0.5:
            starved = [g for g in group if g["chunks_received"] < hi * 0.5]
            ids = ", ".join(str(g["stream_id"]) for g in starved)
            print(
                f"  WARNING: streams [{ids}] received significantly fewer chunks "
                f"({lo}-{min(g['chunks_received'] for g in starved)}) "
                f"vs others ({hi}) with same input ({sent} sent). "
                f"Possible server-side queue drops."
            )

    if output_dir:
        print(f"Output directory: {output_dir}")
    print()

    return clients


def parse_server_uri(server_input):
    """Normalise a server address into a full websocket URI."""
    if server_input.startswith('wss://') or server_input.startswith('ws://'):
        websocket_uri = server_input
        if not websocket_uri.endswith('/v1/realtime'):
            websocket_uri = f"{websocket_uri.rstrip('/')}/v1/realtime"
    else:
        if ':' not in server_input:
            raise ValueError(
                "Server must include port (e.g., 'localhost:9000') or use full URI (e.g., 'wss://grpc.nvcf.nvidia.com')"
            )
        websocket_uri = f"wss://{server_input}/v1/realtime"
    return websocket_uri


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Nemotron Voicechat Client - Stream audio to Nemotron VoiceChat server"
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Server address as 'host:port' or full URI 'wss://host[:port]' (e.g., 'localhost:9000' or 'wss://grpc.nvcf.nvidia.com'). Port is required for host:port format, optional for full URIs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows chunk counts every 50 chunks)",
    )
    parser.add_argument(
        "--no-playback",
        action="store_false",
        dest="playback",
        help="Disable audio playback of server responses (default: enabled)",
    )
    parser.add_argument(
        "--input-file", help="Input audio file to stream instead of microphone (supports WAV)"
    )
    parser.add_argument(
        "--audio-output", help="Output file for received audio (default: received_audio.wav)"
    )
    parser.add_argument(
        "--user-text-output", help="Output file for user ASR text (default: user_text.txt)"
    )
    parser.add_argument(
        "--agent-text-output", help="Output file for agent responses (default: agent_text.txt)"
    )
    parser.add_argument(
        "--function-text-output",
        help="Output file for tool/function call log in JSONL format (default: function_calls.jsonl)",
    )
    parser.add_argument(
        "--conversation-output",
        help="Output file for conversation log in JSONL format (default: conversation.jsonl)",
    )
    parser.add_argument(
        "--api-key", help="API key for NVCF authentication (used with NVIDIA Cloud servers)"
    )
    parser.add_argument(
        "--function-id", help="Function ID for NVCF (used with NVIDIA Cloud servers)"
    )
    parser.add_argument(
        "--instructions", help="Instructions for the agent (inline string or path to a text file)"
    )
    parser.add_argument(
        "--tools", default=None, help="Tools for the agent (JSON string or path to a JSON file)"
    )
    parser.add_argument(
        "--ignore-tool-calls",
        action="store_true",
        help="Log tool calls but do not send any tool response back to the server",
    )

    # Multi-stream options
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=FILE_INPUT_TRAILING_SILENCE_SECONDS,
        help=f"Seconds of silence to append after input file (default: {FILE_INPUT_TRAILING_SILENCE_SECONDS}). Only used with --input-file.",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=1,
        help="Number of concurrent streams to launch (default: 1). Requires --input-file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-stream output files (created if needed). Only used with --num-streams > 1.",
    )

    args = parser.parse_args()

    if args.instructions and os.path.isfile(args.instructions):
        try:
            with open(args.instructions, "r", encoding="utf-8") as f:
                args.instructions = f.read()
        except OSError as e:
            parser.error(f"Could not read instructions file: {e}")

    if args.tools:
        try:
            args.tools = json.loads(args.tools)
        except json.JSONDecodeError:
            try:
                with open(args.tools, encoding="utf-8") as f:
                    args.tools = json.load(f)
            except OSError as e:
                parser.error(f"Could not read tools file: {e}")
        try:
            _validate_tools(args.tools)
        except ValueError as e:
            parser.error(f"Invalid tools spec: {e}")

    try:
        websocket_uri = parse_server_uri(args.server)
    except ValueError as e:
        parser.error(str(e))

    # ---------------------------------------------------------------------- #
    #  Multi-stream mode                                                     #
    # ---------------------------------------------------------------------- #
    if args.num_streams > 1:
        if not args.input_file:
            parser.error(
                "--input-file is required when using --num-streams > 1 (microphone cannot be shared)"
            )

        print("=== Nemotron Voicechat Multi-Stream Client ===")
        print(f"Streams: {args.num_streams}")
        print(f"Input: {args.input_file}")
        print(f"WebSocket: {websocket_uri}")
        if args.output_dir:
            print(f"Output dir: {args.output_dir}")
        print("Press Ctrl+C to stop\n")

        try:
            asyncio.run(
                run_multi_stream(
                    websocket_uri,
                    args.num_streams,
                    args.input_file,
                    verbose=args.verbose,
                    api_key=args.api_key,
                    function_id=args.function_id,
                    instructions=args.instructions,
                    tools=args.tools,
                    output_dir=args.output_dir,
                )
            )
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")

        os._exit(0)

    # ---------------------------------------------------------------------- #
    #  Single-stream mode                                                    #
    # ---------------------------------------------------------------------- #
    print("=== Nemotron Voicechat Client ===")
    print(
        f"Audio config: Input {INPUT_SAMPLE_RATE}Hz ({INPUT_CHUNK_SIZE} samples), Output {OUTPUT_SAMPLE_RATE}Hz ({OUTPUT_CHUNK_SIZE} samples)"
    )
    print(f"Sample rate: {INPUT_SAMPLE_RATE}Hz")
    print(
        f"Chunk duration: Input {INPUT_CHUNK_DURATION_MS}ms, Output {OUTPUT_CHUNK_DURATION_MS}ms, {CHANNELS} channel(s)"
    )
    if args.input_file:
        _, file_rate = sf.read(args.input_file, frames=1, dtype='float32')
        print(f"Input: File ({args.input_file}) [{file_rate}Hz]")
    else:
        print("Input: Microphone")
    print(f"WebSocket: {websocket_uri}")
    print(f"Playback: {'Enabled' if args.playback else 'Disabled'}")
    if not args.input_file:
        print("Session duration limit: 300s (5 minutes)")
    if args.verbose:
        print("Verbose mode enabled")
    print("Press Ctrl+C to stop\n")

    try:
        # Fixed duration of 5 minutes (300 seconds)
        max_duration = 300
        asyncio.run(
            main(
                websocket_uri,
                args.verbose,
                args.audio_output,
                args.user_text_output,
                args.agent_text_output,
                args.function_text_output,
                args.conversation_output,
                args.input_file,
                args.playback,
                max_duration,
                args.api_key,
                args.function_id,
                "pcm16",
                args.instructions,
                args.tools,
                args.ignore_tool_calls,
                args.trailing_silence,
            )
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

    # Always force exit to avoid hanging on thread cleanup
    os._exit(0)
