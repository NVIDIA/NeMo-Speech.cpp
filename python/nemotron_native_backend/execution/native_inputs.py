from __future__ import annotations

from dataclasses import dataclass

from .native_assets import HIDDEN, WIDTHS

MAX_GENERATIONS = 511
RNG_CAPACITY = 64


@dataclass(frozen=True)
class Request:
    text: str
    target_ids: tuple[int, ...]
    context_ids: tuple[int, ...]


def requests_from_texts(assets, texts):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(assets.root / "tokenizer.json"))
    context = tuple(tokenizer.encode("[EN]", add_special_tokens=True).ids)
    if len(context) != 3:
        raise ValueError(f"reviewed context has3tokens; tokenizer returned{context}")
    result = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty already-normalized string")
        ids = tuple(tokenizer.encode(text, add_special_tokens=False).ids) + (131073,)
        if len(ids) > 512:
            raise ValueError("target text exceeds512token state capacity")
        result.append(Request(text, ids, context))
    return result


def validate_requests(requests):
    if len(requests) not in WIDTHS:
        raise ValueError("native exact cohort requires B1/B16/B32/B64")
    if len({len(r.context_ids) for r in requests}) != 1:
        raise ValueError("ragged context lengths are unsupported")
    for r in requests:
        if (
            len(r.context_ids) != 3
            or not 1 <= len(r.target_ids) <= 512
            or r.target_ids[-1] != 131073
        ):
            raise ValueError("invalid native request context/target/EOS contract")
        if any(type(x) is not int or not 0 <= x < 131075 for x in (*r.context_ids, *r.target_ids)):
            raise ValueError("text token outside the checkpoint vocabulary")


class NativeInputs:
    def __init__(self, assets, requests, *, device="cpu"):
        import torch

        validate_requests(requests)
        self.torch = torch
        self.assets = assets
        self.requests = tuple(requests)
        self.batch = len(requests)
        self.device = torch.device(device)
        self.weights = assets.inputs(device=self.device)
        self.offsets = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.text_lengths = torch.tensor(
            [len(r.target_ids) for r in requests], dtype=torch.int64, device=self.device
        )
        self.text_tokens = torch.zeros((self.batch, 512), dtype=torch.int64, device=self.device)
        for i, r in enumerate(requests):
            self.text_tokens[i, : len(r.target_ids)] = torch.tensor(
                r.target_ids, dtype=torch.int64, device=self.device
            )
        self.slots = torch.arange(self.batch, dtype=torch.int64, device=self.device)
        self.last_codes = torch.zeros((self.batch, 16), dtype=torch.int64, device=self.device)
        self.last_phoneme = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.phoneme_ended = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self.audio_valid = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self.stop = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self.control_logits = torch.zeros((self.batch, 2), dtype=torch.bfloat16, device=self.device)
        self.embedding = torch.empty((self.batch, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self.combined_model = torch.empty(
            (self.batch, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        prompts = []
        for r in requests:
            ids = torch.tensor(r.context_ids, dtype=torch.int64, device=self.device)
            prompts.append(
                torch.cat(
                    (self.weights["speaker"], self.weights["text_embedding.weight"][ids]), dim=0
                )
            )
        self.prompt = torch.stack(prompts, dim=1).to(torch.bfloat16).contiguous()
        self.prompt_positions = self.prompt.shape[0]
        self.prepared = False
        self.generation = 0

    def prepare_decode(self):
        torch = self.torch
        if self.prepared:
            raise RuntimeError("previous decode input has not been committed")
        if self.generation >= MAX_GENERATIONS:
            raise RuntimeError("configured generation limit reached")
        (offsets, lengths) = (self.offsets, self.text_lengths)
        safe_offsets = torch.minimum(offsets, torch.clamp_min(lengths - 1, 0))
        text_ids = self.text_tokens[self.slots, safe_offsets]
        combined = self.weights["text_embedding.weight"][text_ids] * (offsets < lengths).unsqueeze(
            -1
        ).to(torch.bfloat16)
        self.audio_valid.copy_(offsets >= 5)
        audio_codes = self.last_codes.clone()
        audio_codes[offsets == 5] = 1024
        audio = self.weights["audio_embeddings.0.weight"][audio_codes[:, 0]]
        for i in range(1, 16):
            audio = audio + self.weights[f"audio_embeddings.{i}.weight"][audio_codes[:, i]]
        audio = audio / 16
        combined = combined + audio * self.audio_valid.unsqueeze(-1).to(audio.dtype)
        phoneme_valid = (offsets >= 3) & ~self.phoneme_ended
        phoneme = self.last_phoneme.clone()
        phoneme[offsets == 3] = 2048
        value = self.weights["phoneme_embeddings.0.weight"][phoneme]
        combined = combined + value * phoneme_valid.unsqueeze(-1).to(value.dtype)
        self.phoneme_ended |= (offsets > 3) & phoneme_valid & (phoneme == 2049)
        self.combined_model.copy_(combined)
        self.embedding.copy_(combined)
        self.prepared = True
        return self.embedding

    def commit_outputs(self, codes, phonemes):
        torch = self.torch
        if not self.prepared:
            raise RuntimeError("commit requires a prepared generation")
        if (
            tuple(codes.shape) != (self.batch, 16)
            or codes.dtype != torch.int64
            or codes.device != self.device
        ):
            raise ValueError("codes must be INT64[B,16] on the owned device")
        if (
            tuple(phonemes.shape) != (self.batch,)
            or phonemes.dtype != torch.int32
            or phonemes.device != self.device
        ):
            raise ValueError("phoneme output must be INT32[B] on the owned device")
        self.stop.copy_((codes == 1025).any(dim=1) & self.audio_valid)
        self.last_codes.copy_(codes)
        self.last_phoneme.copy_(phonemes)
        self.offsets.add_(1)
        self.control_logits.zero_()
        self.control_logits[:, 1] = torch.where(self.stop, 30.0, -30.0).to(torch.bfloat16)
        self.prepared = False
        self.generation += 1
        return self.stop


def cache_plan(batch, *, prompt_positions=67, max_generations=MAX_GENERATIONS):
    if (
        batch not in WIDTHS
        or not 1 <= prompt_positions
        or (not 1 <= max_generations <= MAX_GENERATIONS)
    ):
        raise ValueError("unsupported native cohort/cache capacity")
    max_tokens = prompt_positions + max_generations
    blocks = (max_tokens + 31) // 32
    conv = batch * 3584 * 3 * 2
    ssm = batch * 64 * 24 * 128 * 4
    kv = batch * 2 * blocks * 4 * 32 * 128 * 2
    return {
        "batch": batch,
        "prompt_positions": prompt_positions,
        "max_generations": max_generations,
        "max_sequence_tokens": max_tokens,
        "tokens_per_block": 32,
        "blocks_per_sequence": blocks,
        "conv_shape_per_layer": [batch, 3584, 3],
        "conv_dtype": "BF16",
        "conv_bytes_per_layer": conv,
        "ssm_shape_per_layer": [batch, 64, 24, 128],
        "ssm_dtype": "FP32",
        "ssm_bytes_per_layer": ssm,
        "kv_shape_per_layer": [batch * 2 * blocks, 4, 32, 128],
        "kv_dtype": "BF16",
        "kv_bytes_per_layer": kv,
        "block_table_shape_per_layer": [batch, 2, blocks],
        "block_table_dtype": "INT32",
        "total_recurrent_payload_bytes": 14 * (conv + ssm) + 3 * kv,
        "slot_map": "identity INT32[batch]; no reassignment while cohort is active",
        "initial_past_len": prompt_positions,
        "past_len_update": "mk2_run_native_stage0 increments device lengths; Python must not "
        "increment",
        "ownership": "native prefill handle; separate Stage0 handle borrows stable cache views",
    }
