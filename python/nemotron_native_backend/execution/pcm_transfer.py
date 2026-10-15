def install(engine):
    import torch

    from .. import r31

    codec = engine.core.codec
    stream = torch.cuda.Stream(device=codec.device)
    device_finite = r31.enabled("fast_first_chunk")
    codec.device_finite = device_finite
    slots = []
    for _ in range(2):
        slots.append(
            {
                "pcm": torch.empty(engine.batch * 8 * 1764, dtype=torch.float32, pin_memory=True),
                "metadata": torch.empty(engine.batch + 4, dtype=torch.int32, pin_memory=True),
                "finite": torch.empty(1, dtype=torch.bool, pin_memory=True),
                "done": torch.cuda.Event(),
                "ticket": None,
            }
        )
    submit = engine._submit_chunk
    finish = codec.finish
    close = engine.close
    cursor = 0
    pending = {}

    def submit_ticket(ticket, *, row_count=None):
        nonlocal cursor
        with codec._lock:
            if codec._ticket is not ticket or ticket.owner is not codec:
                raise RuntimeError("pinned ticket ownership mismatch")
            if id(ticket) in pending:
                raise RuntimeError("pinned ticket already submitted")
            if row_count is not None and (
                type(row_count) is not int or not 0 <= row_count <= engine.batch
            ):
                raise ValueError("PCM transfer row count must fit the physical batch")
            slot = slots[cursor]
            if slot["ticket"] is not None:
                raise RuntimeError("pinned PCM lease reused before release")
            slot["ticket"] = ticket
            try:
                source = (
                    ticket.pcm[:row_count]
                    if row_count is not None and device_finite
                    else ticket.pcm
                )
                pcm = slot["pcm"][: source.numel()].view_as(source)
                stream.wait_event(ticket.event)
                with torch.cuda.stream(stream):
                    if source.numel():
                        pcm.copy_(source, non_blocking=True)
                    slot["metadata"][:4].copy_(ticket.status, non_blocking=True)
                    slot["metadata"][4:].copy_(ticket.samples, non_blocking=True)
                    if device_finite:
                        if ticket.finite is None:
                            raise RuntimeError(
                                "device finiteness flag missing from the codec ticket"
                            )
                        slot["finite"].copy_(ticket.finite, non_blocking=True)
                    slot["done"].record(stream)
                pending[id(ticket)] = (slot, pcm)
                cursor = 1 - cursor
            except BaseException:
                codec._poisoned = True
                raise

    def submit_chunk(*args, **kwargs):
        result = submit(*args, **kwargs)
        submit_ticket(result["ticket"])
        return result

    def finish_ticket(ticket, copy_pcm=False):
        item = pending.get(id(ticket))
        if item is None:
            return finish(ticket, copy_pcm=copy_pcm)
        (slot, pcm) = item
        with codec._lock:
            if codec._ticket is not ticket or ticket.owner is not codec:
                raise RuntimeError("pinned ticket ownership mismatch")
            try:
                slot["done"].synchronize()
                status = slot["metadata"][:4].tolist()
                if status[0]:
                    raise RuntimeError("codec validation failed: " + str(status))
                result = {
                    "pcm": pcm if copy_pcm else ticket.pcm,
                    "next_states": ticket.next_states,
                    "samples": slot["metadata"][4:],
                    "status": status,
                    "sample_rate_hz": 22050,
                }
                if device_finite and copy_pcm:
                    result["finite"] = bool(slot["finite"][0])
                return result
            except BaseException:
                codec._poisoned = True
                raise
            finally:
                codec._ticket = None
                slot["ticket"] = None
                pending.pop(id(ticket), None)

    def close_engine():
        stream.synchronize()
        return close()

    engine._submit_chunk = submit_chunk
    engine.submit_pcm_transfer = submit_ticket
    codec.finish = finish_ticket
    engine.close = close_engine
    engine._r8_pcm_owners = (stream, slots, pending)
