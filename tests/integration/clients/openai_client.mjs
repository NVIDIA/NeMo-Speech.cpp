#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test the standard OpenAI JavaScript client against a local NeMo-Speech.cpp server.

import { createReadStream } from "node:fs";
import { writeFile } from "node:fs/promises";
import OpenAI from "openai";

const [audioPath, speechText] = process.argv.slice(2);
if (!audioPath) {
  console.error("Usage: node openai_client.mjs AUDIO.wav [TEXT_TO_SPEAK]");
  process.exit(2);
}

const client = new OpenAI({
  baseURL: process.env.OPENAI_BASE_URL ?? "http://127.0.0.1:8080/v1",
  apiKey: process.env.NEMO_SPEECH_API_KEY ?? "local",
});

const transcript = await client.audio.transcriptions.create({
  model: "default",
  file: createReadStream(audioPath),
  response_format: "verbose_json",
});
console.log(transcript.text);

if (speechText) {
  const response = await client.audio.speech.create({
    model: "default",
    voice: "default",
    input: speechText,
    response_format: "wav",
  });
  await writeFile("speech.wav", Buffer.from(await response.arrayBuffer()));
  console.log("Wrote speech.wav");
}
