#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump NeMo Sortformer V2/V3 reference tensors for ggml parity tests.

With --steady-state, capture cold, FIFO-only, and full-cache neural steps for
either version. Geometry and array shapes are stored in the same NERB-compatible
NPZ format as the V2 streaming reference; no external fixture generator is needed.

Without --steady-state, replays the V2 sync streaming loop (forward_streaming with
async_streaming=False) chunk by chunk, capturing every intermediate the C++
port needs to check against:

  per chunk: mel window, pre_encode embeddings, concat lengths, full preds,
             chunk preds, post-update AOSC state (spkcache/preds/fifo/
             mean_sil_emb/n_sil_frames), compression flag.
  stream:    full mel sequence, final concatenated chunk preds.

Default geometry uses the V2 runtime streaming preset; pass V3 geometry explicitly.
Deep intermediates (fc-encoder output, transformer output) are stored for the first
--deep-chunks chunks and for every compression chunk.

Usage:
    python dump_sortformer_reference.py <ckpt.nemo> <audio.wav> <out.npz> \
        [--device cpu] [--max-sec 40] [--chunk 8 --lc 0 --rc 8 \
         --fifo 80 --spkcache 160 --update-period 80]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def dump_neural_steps(model, mel, args):
    """Capture exact-length neural boundaries independently of the state update."""
    sub = model.encoder.subsampling_factor
    frames = (args.chunk + args.lc + args.rc) * sub
    if mel.shape[-1] < frames:
        raise ValueError(f"audio must provide at least {frames} mel frames")
    chunk = mel[:, :, :frames].transpose(1, 2).contiguous()
    lengths = torch.tensor([frames], device=mel.device, dtype=torch.int64)
    embeddings, embedding_lengths = model._call_pre_encode(chunk, lengths)
    dim = embeddings.shape[-1]
    rng = torch.Generator(device=mel.device).manual_seed(0)
    output = {
        "n_cases": np.array([3], dtype=np.int64),
        "output_factor": np.array([model.upsample_factor], dtype=np.int64),
        "geometry": np.array(
            [args.chunk, args.lc, args.rc, args.fifo, args.spkcache, args.update_period],
            dtype=np.int64,
        ),
    }
    for index, (cache_frames, fifo_frames) in enumerate(
        [(0, 0), (0, args.fifo), (args.spkcache, args.fifo)]
    ):
        cache = torch.randn((1, cache_frames, dim), device=mel.device, generator=rng) * 0.01
        fifo = torch.randn((1, fifo_frames, dim), device=mel.device, generator=rng) * 0.01
        packed, packed_lengths = model.sortformer_modules.concat_and_pad(
            [cache, fifo, embeddings],
            [
                lengths.new_tensor([cache_frames]),
                lengths.new_tensor([fifo_frames]),
                embedding_lengths,
            ],
            output_length=cache_frames + fifo_frames + embeddings.shape[1],
        )
        encoded, encoded_lengths = model.frontend_encoder(
            processed_signal=packed, processed_signal_length=packed_lengths, bypass_pre_encode=True
        )
        native = model.forward_infer(encoded, encoded_lengths)
        coarse = (
            model.sortformer_modules.downsample_preds(native, model.upsample_factor)
            if model.high_resolution
            else native
        )
        prefix = f"case{index:03d}/"
        for name, tensor in {
            "mel_window": chunk,
            "spkcache": cache,
            "fifo": fifo,
            "pre_encode": embeddings,
            "preds_full": coarse,
            "preds_native": native,
        }.items():
            output[prefix + name] = tensor[0].float().cpu().numpy()
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help=".nemo checkpoint path")
    ap.add_argument("audio", help="16 kHz mono wav")
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--max-sec", type=float, default=40.0)
    ap.add_argument(
        "--steady-state", action="store_true", help="V2/V3 cold, FIFO, and full-cache graph parity"
    )
    # Defaults = the runtime streaming preset (DiarGeometry, aosc_state.h) for
    # convenience only: the chosen geometry is recorded in the dump and the
    # parity test reads it from there, so any values make a valid reference.
    ap.add_argument("--chunk", type=int, default=20, help="chunk_len in 80ms encoder frames")
    ap.add_argument("--lc", type=int, default=0, help="chunk_left_context in encoder frames")
    ap.add_argument("--rc", type=int, default=0, help="chunk_right_context in encoder frames")
    ap.add_argument("--fifo", type=int, default=80)
    ap.add_argument("--spkcache", type=int, default=160)
    ap.add_argument("--update-period", type=int, default=80)
    ap.add_argument(
        "--deep-chunks",
        type=int,
        default=3,
        help="store fc-encoder/transformer outputs for the first N chunks",
    )
    args = ap.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel

    # Reference tensors must not inherit a container's TF32 defaults.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = SortformerEncLabelModel.restore_from(restore_path=args.ckpt, map_location=device)
    model.eval()
    model.to(device)
    if model.high_resolution and not args.steady_state:
        ap.error("V3 reference generation requires --steady-state; the full-stream dump is V2-only")

    sm = model.sortformer_modules
    sm.chunk_len = args.chunk
    sm.chunk_left_context = args.lc
    sm.chunk_right_context = args.rc
    sm.fifo_len = args.fifo
    sm.spkcache_len = args.spkcache
    sm.spkcache_update_period = args.update_period
    sm._check_streaming_parameters()

    # Determinism: no dither (riva sets 0 at inference too).
    model.preprocessor.featurizer.dither = 0.0

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert sr == 16000, f"expected 16 kHz audio, got {sr}"
    max_samples = int(args.max_sec * sr)
    audio = audio[:max_samples]

    sig = torch.from_numpy(audio).unsqueeze(0).to(device)
    sig_len = torch.tensor([sig.shape[1]], device=device)

    with torch.inference_mode():
        # Streaming mode: no max-normalization (SortformerEncLabelModel.
        # process_signal only rescales when streaming_mode is off).
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)

        if args.steady_state:
            output = dump_neural_steps(model, mel[:, :, : int(mel_len[0])], args)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out, **output)
            print(f"[dump] 3 neural-step cases -> {args.out}")
            return 0

        out: dict[str, np.ndarray] = {
            "audio": audio,
            "mel": mel[0].cpu().numpy(),  # (128, T)
            "mel_len": mel_len.cpu().numpy(),
            "geometry": np.array(
                [args.chunk, args.lc, args.rc, args.fifo, args.spkcache, args.update_period],
                dtype=np.int64,
            ),
        }

        state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)
        total_preds = torch.zeros((1, 0, sm.n_spk), device=device)
        offset = torch.zeros((1,), dtype=torch.long, device=device)
        sub = model.encoder.subsampling_factor

        n_chunks = 0
        compression_chunks = []
        for idx, chunk_feat, feat_lengths, left_off, right_off in sm.streaming_feat_loader(
            feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=offset
        ):
            p = f"chunk{idx:03d}/"
            out[p + "mel_window"] = chunk_feat[0].cpu().numpy()  # (T_mel, 128)
            out[p + "feat_length"] = feat_lengths.cpu().numpy()
            out[p + "offsets"] = np.array([left_off, right_off], dtype=np.int64)

            pre_embs, pre_lens = model.encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)
            out[p + "pre_encode"] = pre_embs[0].cpu().numpy()  # (T_enc, 512)

            spk_len_before = state.spkcache.shape[1]
            fifo_len_before = state.fifo.shape[1]
            out[p + "state_lens_before"] = np.array(
                [spk_len_before, fifo_len_before, int(pre_lens[0])], dtype=np.int64
            )

            concat_embs = sm.concat_embs(
                [state.spkcache, state.fifo, pre_embs], dim=1, device=device
            )
            concat_lens = spk_len_before + fifo_len_before + pre_lens

            fc_embs, fc_lens = model.frontend_encoder(
                processed_signal=concat_embs,
                processed_signal_length=concat_lens,
                bypass_pre_encode=True,
            )
            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)
            preds = sm.apply_mask_to_preds(preds, fc_lens)
            out[p + "preds_full"] = preds[0].cpu().numpy()  # (L1+L2+L3, 4)

            deep = idx < args.deep_chunks
            if deep:
                # frontend_encoder applies encoder_proj, so this is the
                # post-projection (L, 192) input to the transformer stack.
                out[p + "fc_encoder"] = fc_embs[0].cpu().numpy()
            lc = round(left_off / sub)
            rc = math.ceil(right_off / sub)

            state, chunk_preds = sm.streaming_update(
                streaming_state=state, chunk=pre_embs, preds=preds, lc=lc, rc=rc
            )
            total_preds = torch.cat([total_preds, chunk_preds], dim=1)

            out[p + "chunk_preds"] = chunk_preds[0].cpu().numpy()
            out[p + "spkcache_after"] = state.spkcache[0].cpu().numpy()
            if state.spkcache_preds is not None:
                out[p + "spkcache_preds_after"] = state.spkcache_preds[0].cpu().numpy()
            out[p + "fifo_after"] = state.fifo[0].cpu().numpy()
            out[p + "mean_sil_emb_after"] = state.mean_sil_emb[0].cpu().numpy()
            out[p + "n_sil_frames_after"] = state.n_sil_frames.cpu().numpy()
            # Compression happened iff the cache is back at its cap after
            # having exceeded it (pre-update len + pop > cap).
            if (
                state.spkcache.shape[1] == sm.spkcache_len
                and spk_len_before + (fifo_len_before + chunk_preds.shape[1] - state.fifo.shape[1])
                > sm.spkcache_len
            ):
                compression_chunks.append(idx)
            n_chunks = idx + 1

        out["total_preds"] = total_preds[0].cpu().numpy()
        out["n_chunks"] = np.array([n_chunks], dtype=np.int64)
        out["compression_chunks"] = np.array(compression_chunks, dtype=np.int64)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    total_mb = sum(v.nbytes for v in out.values()) / 1e6
    print(
        f"[dump] {n_chunks} chunks, compressions at {compression_chunks}, "
        f"{total_mb:.1f} MB raw -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
