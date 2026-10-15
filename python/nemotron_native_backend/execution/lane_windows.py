from __future__ import annotations


class LaneWindow:
    def __init__(self, batch, frames, *, done, device="cuda:0", raw=None, stops=None):
        import torch

        if (
            batch not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
            or not 1 <= frames <= 16
        ):
            raise ValueError("invalid window geometry")
        self.torch = torch
        self.batch = batch
        self.frames = frames
        (B, F) = (batch, frames)
        device = torch.device(device)
        if (
            done.dtype != torch.bool
            or tuple(done.shape) != (B,)
            or done.device != device
            or (not done.is_contiguous())
        ):
            raise ValueError("done must be contiguous bool[B] on the window device")
        self.done = done

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self.raw = empty((B, F, 16), torch.int64) if raw is None else raw
        self.stops = empty((B, F), torch.bool) if stops is None else stops
        if (
            tuple(self.raw.shape) != (B, F, 16)
            or tuple(self.stops.shape) != (B, F)
            or self.raw.dtype != torch.int64
            or (self.stops.dtype != torch.bool)
            or (self.raw.device != device)
            or (self.stops.device != device)
        ):
            raise ValueError("borrowed history shape/dtype/device mismatch")
        self.packed = empty((B, F, 16), torch.int64)
        self.lengths = empty((B,), torch.int32)
        self.element_ok = empty((B, F, 16), torch.bool)
        self.element_tmp = empty((B, F, 16), torch.bool)
        self.valid = empty((B, F), torch.bool)
        self.nonzero = empty((B, F), torch.bool)
        self.stop_prefix = empty((B, F), torch.int32)
        self.stop_prefix_before = empty((B, F), torch.int32)
        self.before_stop = empty((B, F), torch.bool)
        self.was_active = empty((B,), torch.bool)
        self.window_eos = empty((B,), torch.bool)
        self.rank = empty((B, F), torch.int64)
        self.matches = empty((B, F, F), torch.bool)
        self.has_match = empty((B, F), torch.bool)
        self.source_index = empty((B, F), torch.int64)
        self.targets = torch.arange(1, F + 1, dtype=torch.int64, device=device).view(1, F, 1)
        self._rank3 = self.rank.view(B, 1, F)
        self._valid3 = self.valid.view(B, 1, F)
        self._active2 = self.was_active.view(B, 1)
        self._gather_index = self.source_index.unsqueeze(-1).expand(B, F, 16)
        self._has_match3 = self.has_match.unsqueeze(-1)
        self.raw_columns = tuple(self.raw[:, i, :] for i in range(F))
        self.stop_columns = tuple(self.stops[:, i] for i in range(F))
        self.fill = 0

    def append(self, codes, stop):
        if self.fill >= self.frames:
            raise RuntimeError("window capacity exceeded")
        self.raw_columns[self.fill].copy_(codes)
        self.stop_columns[self.fill].copy_(stop)
        self.fill += 1

    def pack(self):
        if self.fill != self.frames:
            raise RuntimeError("window must be full before packing")
        t = self.torch
        t.ge(self.raw, 0, out=self.element_ok)
        t.lt(self.raw, 1024, out=self.element_tmp)
        t.logical_and(self.element_ok, self.element_tmp, out=self.element_ok)
        t.all(self.element_ok, dim=2, out=self.valid)
        t.ne(self.raw, 0, out=self.element_tmp)
        t.any(self.element_tmp, dim=2, out=self.nonzero)
        t.logical_and(self.valid, self.nonzero, out=self.valid)
        t.logical_not(self.done, out=self.was_active)
        t.cumsum(self.stops, dim=1, dtype=t.int32, out=self.stop_prefix)
        self.stop_prefix_before[:, :1].zero_()
        self.stop_prefix_before[:, 1:].copy_(self.stop_prefix[:, :-1])
        t.eq(self.stop_prefix_before, 0, out=self.before_stop)
        t.logical_and(self.before_stop, self._active2, out=self.before_stop)
        t.logical_and(self.valid, self.before_stop, out=self.valid)
        t.cumsum(self.valid, dim=1, dtype=t.int64, out=self.rank)
        t.eq(self._rank3, self.targets, out=self.matches)
        t.logical_and(self.matches, self._valid3, out=self.matches)
        t.max(self.matches, dim=2, out=(self.has_match, self.source_index))
        t.gather(self.raw, 1, self._gather_index, out=self.packed)
        self.packed.mul_(self._has_match3)
        t.sum(self.valid, dim=1, dtype=t.int32, out=self.lengths)
        t.any(self.stops, dim=1, out=self.window_eos)
        t.logical_or(self.done, self.window_eos, out=self.done)
        return (self.packed, self.lengths)
