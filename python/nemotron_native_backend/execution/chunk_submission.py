def install(e):
    t = e.t
    c = e.core
    n = c.neural
    remaining = t.empty_like(e.counts)

    def submit(F, counts, bank):
        if e._codec_pending is not None:
            raise RuntimeError("codec buffer/scratch lease still outstanding")
        (packed, lengths, index, offsets, mask, _, _, banks) = e.bindings[F]
        (ready, a, b) = e._event_pool[len(e._chunk_events)]
        sizes = [
            min(F, available - used)
            for (available, used) in zip(counts, e.consumed_host, strict=False)
        ]
        with t.cuda.stream(n.stream):
            t.sub(e.counts, e.consumed, out=remaining)
            remaining.clamp_(min=0, max=F)
            lengths.copy_(remaining)
            t.add(e.consumed.view(e.batch, 1), offsets, out=index)
            index.clamp_(max=e.steps - 1)
            t.gather(e.queue, 1, index.unsqueeze(-1).expand(e.batch, F, 16), out=packed)
            t.lt(offsets, lengths.view(e.batch, 1), out=mask)
            packed.mul_(mask.unsqueeze(-1))
            e.consumed.add_(lengths)
            ready.record(n.stream)
        e.consumed_host = [
            used + size for (used, size) in zip(e.consumed_host, sizes, strict=False)
        ]
        e.codec_stream.wait_event(ready)
        with t.cuda.stream(e.codec_stream):
            a.record(e.codec_stream)
            banks[bank].event = b
            ticket = banks[bank].enqueue()
        pending = dict(ticket=ticket, sizes=sizes, events=(a, b))
        e._codec_pending = pending
        e._chunk_events.append((a, b))
        return pending

    original = e._submit_chunk
    if hasattr(e, "_r8_pcm_owners"):
        raise RuntimeError("chunk binding must precede pinned transfer binding")
    e._submit_chunk = submit
    e._r9_chunk_owners = (remaining, original)
