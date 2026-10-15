from __future__ import annotations

from .native_inputs import MAX_GENERATIONS, NativeInputs


class ReusableInputs(NativeInputs):
    @classmethod
    def adopt(cls, existing):
        obj = object.__new__(cls)
        obj.__dict__ = existing.__dict__.copy()
        obj._allocate_scratch()
        return obj

    def _allocate_scratch(self):
        torch = self.torch
        B = self.batch

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.max_text_offsets = (self.text_lengths - 1).clamp_min_(0)
        self.safe_offsets = empty((B,), torch.int64)
        self.text_ids = empty((B, 1), torch.int64)
        self.text_mask = empty((B,), torch.bool)
        self.bos_mask = empty((B,), torch.bool)
        self.audio_codes = empty((B, 16), torch.int64)
        self.audio_indices = empty((B, 16), torch.int64)
        self.audio_offsets = (
            torch.arange(16, dtype=torch.int64, device=self.device).reshape(1, 16) * 1032
        )
        self.audio_bos = torch.tensor(1024, dtype=torch.int64, device=self.device)
        self.audio_table = torch.cat(
            [self.weights[f"audio_embeddings.{i}.weight"] for i in range(16)], dim=0
        ).contiguous()
        self.audio_values = empty((B * 16, 1536), torch.bfloat16)
        self.audio_frame = empty((B, 1536), torch.bfloat16)
        self.phoneme_valid = empty((B,), torch.bool)
        self.not_ended = empty((B,), torch.bool)
        self.phoneme_ids = empty((B,), torch.int64)
        self.phoneme_bos = torch.tensor(2048, dtype=torch.int64, device=self.device)
        self.phoneme_value = empty((B, 1536), torch.bfloat16)
        self.feed_eos = empty((B,), torch.bool)
        self.after_phoneme_delay = empty((B,), torch.bool)
        self.code_eos = empty((B, 16), torch.bool)
        self.any_eos = empty((B,), torch.bool)
        self.positive_logit = torch.tensor(30.0, dtype=torch.bfloat16, device=self.device)
        self.negative_logit = torch.tensor(-30.0, dtype=torch.bfloat16, device=self.device)
        self._audio_values3 = self.audio_values.view(B, 16, 1536)
        self._audio_values_rows = tuple(self._audio_values3[:, i, :] for i in range(16))
        self._audio_index_flat = self.audio_indices.view(-1)
        self._text_ids_flat = self.text_ids.view(-1)
        self._safe_offsets_column = self.safe_offsets.view(B, 1)
        self._text_mask_column = self.text_mask.view(B, 1)
        self._bos_mask_column = self.bos_mask.view(B, 1)
        self._audio_valid_column = self.audio_valid.view(B, 1)
        self._phoneme_valid_column = self.phoneme_valid.view(B, 1)
        self._stop_logit = self.control_logits[:, 1]
        self._continue_logit = self.control_logits[:, 0]

    def prepare_decode(self):
        torch = self.torch
        if self.prepared or self.generation >= MAX_GENERATIONS:
            raise RuntimeError(
                "prepare requires uncommitted-free state and remaining generation capacity"
            )
        torch.minimum(self.offsets, self.max_text_offsets, out=self.safe_offsets)
        torch.gather(self.text_tokens, 1, self._safe_offsets_column, out=self.text_ids)
        torch.index_select(
            self.weights["text_embedding.weight"], 0, self._text_ids_flat, out=self.combined_model
        )
        torch.lt(self.offsets, self.text_lengths, out=self.text_mask)
        self.combined_model.mul_(self._text_mask_column)
        torch.ge(self.offsets, 5, out=self.audio_valid)
        torch.eq(self.offsets, 5, out=self.bos_mask)
        torch.where(self._bos_mask_column, self.audio_bos, self.last_codes, out=self.audio_codes)
        torch.add(self.audio_codes, self.audio_offsets, out=self.audio_indices)
        torch.index_select(self.audio_table, 0, self._audio_index_flat, out=self.audio_values)
        self.audio_frame.copy_(self._audio_values_rows[0])
        for value in self._audio_values_rows[1:]:
            torch.add(self.audio_frame, value, out=self.audio_frame)
        self.audio_frame.div_(16)
        self.audio_frame.mul_(self._audio_valid_column)
        torch.add(self.combined_model, self.audio_frame, out=self.combined_model)
        torch.ge(self.offsets, 3, out=self.phoneme_valid)
        torch.logical_not(self.phoneme_ended, out=self.not_ended)
        torch.logical_and(self.phoneme_valid, self.not_ended, out=self.phoneme_valid)
        torch.eq(self.offsets, 3, out=self.bos_mask)
        torch.where(self.bos_mask, self.phoneme_bos, self.last_phoneme, out=self.phoneme_ids)
        torch.index_select(
            self.weights["phoneme_embeddings.0.weight"], 0, self.phoneme_ids, out=self.phoneme_value
        )
        self.phoneme_value.mul_(self._phoneme_valid_column)
        torch.add(self.combined_model, self.phoneme_value, out=self.combined_model)
        torch.eq(self.phoneme_ids, 2049, out=self.feed_eos)
        torch.gt(self.offsets, 3, out=self.after_phoneme_delay)
        torch.logical_and(self.feed_eos, self.after_phoneme_delay, out=self.feed_eos)
        torch.logical_and(self.feed_eos, self.phoneme_valid, out=self.feed_eos)
        torch.logical_or(self.phoneme_ended, self.feed_eos, out=self.phoneme_ended)
        self.embedding.copy_(self.combined_model)
        self.prepared = True
        return self.embedding

    def commit_outputs(self, codes, phonemes):
        torch = self.torch
        if not self.prepared:
            raise RuntimeError("commit requires prepared inputs")
        if (
            codes.dtype != torch.int64
            or tuple(codes.shape) != (self.batch, 16)
            or codes.device != self.device
        ):
            raise ValueError("codes ABI mismatch")
        if (
            phonemes.dtype != torch.int32
            or tuple(phonemes.shape) != (self.batch,)
            or phonemes.device != self.device
        ):
            raise ValueError("phoneme ABI mismatch")
        torch.eq(codes, 1025, out=self.code_eos)
        torch.any(self.code_eos, dim=1, out=self.any_eos)
        torch.logical_and(self.any_eos, self.audio_valid, out=self.stop)
        self.last_codes.copy_(codes)
        self.last_phoneme.copy_(phonemes)
        self.offsets.add_(1)
        self._continue_logit.zero_()
        torch.where(self.stop, self.positive_logit, self.negative_logit, out=self._stop_logit)
        self.prepared = False
        self.generation += 1
        return self.stop
