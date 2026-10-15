import operator


def effective_seed(seed, epoch):
    if type(seed) is not int or not 0 <= seed <= 9223372036854775807:
        raise ValueError("seed must be an unsigned 63-bit integer")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    return seed + epoch - 1 & 9223372036854775807


def generation_deficit(counts, consumed, done, target, executed, limit):
    if not len(counts) == len(consumed) == len(done):
        raise ValueError("queue metadata widths differ")
    if target not in (3, 5, 8) or not 0 <= executed <= limit:
        raise ValueError("invalid generation/chunk bounds")
    if True in done:
        pending = [
            n - used
            for (n, used, terminal) in zip(counts, consumed, done, strict=False)
            if not terminal
        ]
        need = max(0, target - min(pending)) if pending else 0
    else:
        need = max(0, target - min(map(operator.sub, counts, consumed)))
    return min(need, limit - executed)


def continuation_lanes(counts, consumed, held):
    return [
        lane
        for (lane, (count, used, pending)) in enumerate(zip(counts, consumed, held, strict=False))
        if pending and count > used
    ]


def publication_probe_steps(needed, held, done):
    return (
        min(needed, 1)
        if any(pending and (not terminal) for (pending, terminal) in zip(held, done, strict=False))
        else needed
    )


def validate_execution(row):
    mode = row.get("mode", "serial")
    policy = row.get("grid_policy", "static")
    if mode not in ("serial", "overlap", "after-first-publication"):
        raise ValueError("unsupported overlap mode")
    if policy not in ("static", "overlap-only"):
        raise ValueError("unsupported grid policy")
    out = dict(
        mode=mode,
        grid_policy=policy,
        stage0_grid=row.get("stage0_grid", 0),
        codec_grid=row.get("codec_grid", 0),
    )
    for key in ("stage0_grid", "codec_grid"):
        if type(out[key]) is not int or out[key] < 0:
            raise ValueError("grid must be a nonnegative integer")
    if mode != "serial" and (not out["stage0_grid"] or not out["codec_grid"]):
        raise ValueError("overlap requires explicit bounded grids")
    if policy == "overlap-only":
        full = row.get("full_stage0_grid")
        if type(full) is not int or full < out["stage0_grid"] or full < 1:
            raise ValueError("overlap-only requires a full grid at least as large as overlap grid")
        out["full_stage0_grid"] = full
    return out


def should_prefetch(
    mode, counts, consumed, done, held, sizes, first_published, next_frames, executed, limit
):
    if mode == "serial" or (mode == "after-first-publication" and (not all(first_published))):
        return False
    future_held = [pending or size > 0 for (pending, size) in zip(held, sizes, strict=False)]
    for lane in continuation_lanes(counts, consumed, future_held):
        future_held[lane] = False
    needed = generation_deficit(counts, consumed, done, next_frames, executed, limit)
    return publication_probe_steps(needed, future_held, done) == 1


def select_retained_rows(done, target):
    if not done or type(target) is not int or (not 0 < target < len(done)):
        raise ValueError("invalid width contraction")
    if any(type(v) is not bool for v in done):
        raise ValueError("done must contain host booleans")
    live = [i for (i, v) in enumerate(done) if not v]
    if not 0 < len(live) <= target:
        return None
    return live + [i for (i, v) in enumerate(done) if v][: target - len(live)]


def validate_active_lanes(row, batch, execution):
    if row is None or row.get("mode") == "disabled":
        return None
    if row.get("mode") != "fixed-subset" or row.get("target_batch") != 16 or batch not in (32, 64):
        raise ValueError("contraction supports B32/B64 to B16 only")
    if row.get("after_chunk0") is not True:
        raise ValueError("contraction must preserve the chunk0 path")
    grid = row.get("narrow_grid")
    if type(grid) is not int or grid <= 0:
        raise ValueError("narrow grid must be a positive integer")
    policy = row.get("grid_policy", "static")
    if policy not in ("static", "follow-stage0"):
        raise ValueError("unsupported narrow grid policy")
    if policy == "follow-stage0" and (
        execution.get("grid_policy") != "overlap-only" or grid != execution.get("stage0_grid")
    ):
        raise ValueError(
            "dynamic narrow grid must follow an explicit bounded Stage0 execution grid"
        )
    for name in ("stage0", "transfer"):
        artifact = row.get(name, {})
        if (
            set(artifact) != {"path", "sha256"}
            or not isinstance(artifact["path"], str)
            or len(artifact["sha256"]) != 64
        ):
            raise ValueError("auxiliary library requires exact path and SHA256")
    return dict(row)
