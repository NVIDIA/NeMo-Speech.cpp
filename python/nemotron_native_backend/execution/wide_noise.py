from __future__ import annotations

MAX_SEED = 9223372036854775807
GROUP_STRIDE = 521288629


def group_seed(effective_seed, group):
    if (
        type(effective_seed) is not int
        or not 0 <= effective_seed <= MAX_SEED
        or type(group) is not int
        or (group < 0)
    ):
        raise ValueError("invalid effective seed or RNG group")
    return effective_seed + group * GROUP_STRIDE & MAX_SEED


class CohortNoise:
    def __init__(self, *, device="cpu", base_seed=0, epoch=1, batch=64):
        import torch

        if (
            batch not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
            or not 0 <= base_seed <= MAX_SEED
            or epoch < 1
        ):
            raise ValueError("unsupported batch/seed/epoch")
        self.torch = torch
        self.device = torch.device(device)
        self.batch = batch
        self.buffer = torch.empty((max(64, batch), 16, 80), dtype=torch.float32, device=self.device)
        self.groups = tuple(self.buffer[i : i + 64] for i in range(0, max(64, batch), 64))
        self.generators = tuple(torch.Generator(device=self.device) for _ in self.groups)
        self.generator = self.generators[0]
        self._ring = None
        self._predrawn = None
        self._replay = 0
        self.predraw_hits = 0
        self.reset(base_seed + epoch - 1 & MAX_SEED)

    def reset(self, seed):
        self.effective_seed = seed
        self.draws = 0
        self.context_drawn = False
        if self._predrawn is not None and self._predrawn[0] == seed:
            self._replay = self._predrawn[1]
            self._predrawn = None
            self.predraw_hits += 1
            return
        self._predrawn = None
        self._replay = 0
        for group, generator in enumerate(self.generators):
            generator.manual_seed(group_seed(seed, group))

    def predraw(self, seed, count):
        torch = self.torch
        if self._ring is None or self._ring.shape[0] < count:
            self._ring = torch.empty(
                (count, *self.buffer.shape), dtype=self.buffer.dtype, device=self.device
            )
        self._predrawn = None
        self._replay = 0
        for group, generator in enumerate(self.generators):
            generator.manual_seed(group_seed(seed, group))
        self._draw_into_buffer()
        for index in range(count):
            self._draw_into_buffer()
            self._ring[index].copy_(self.buffer)
        self._predrawn = (seed, count)

    def _draw_into_buffer(self):
        for value, generator in zip(self.groups, self.generators, strict=False):
            value.exponential_(generator=generator)
            value.log_().neg_()

    def _draw(self):
        self._draw_into_buffer()
        self.draws += 1

    def context(self):
        if self.context_drawn:
            raise RuntimeError("context RNG draw occurs exactly once")
        if self._replay:
            self.draws += 1
        else:
            self._draw()
        self.context_drawn = True

    def generation(self, batch):
        if not self.context_drawn or batch != self.batch:
            raise RuntimeError("RNG cohort width/context mismatch")
        index = self.draws - 1
        if index < self._replay:
            self.buffer.copy_(self._ring[index])
            self.draws += 1
        else:
            self._replay = 0
            self._draw()
        return self.buffer[:batch]

    def provenance(self):
        result = {
            "algorithm": "torch.exponential_(generator).log_().neg_()",
            "capacity_shape": list(self.buffer.shape),
            "effective_seed": self.effective_seed,
            "device": str(self.device),
            "draws": self.draws,
            "context_discarded_draws": int(self.context_drawn),
            "scope": "CPU and CUDA generator streams are not bit-identical. Exact CUDA stream "
            "equivalence requires GPU qualification.",
        }
        if self.batch > 64:
            result.update(
                schema="native.rng-independent64.v1",
                group_rows=64,
                group_seeds=[
                    group_seed(self.effective_seed, g) for g in range(len(self.generators))
                ],
                group_stride=GROUP_STRIDE,
                dead_lanes="Every group advances through every physical generation, including "
                "finished lanes.",
            )
        return result
