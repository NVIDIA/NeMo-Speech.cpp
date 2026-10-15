def pack_and_enqueue(window, binding, start_event, stream):
    window.pack()
    start_event.record(stream)
    return binding.enqueue()


def finish_and_observe(codec, ticket, window, done):
    result = codec.finish(ticket, copy_pcm=True)
    return (result, window.lengths.cpu().tolist(), window.stops.cpu().tolist(), done.cpu().tolist())
