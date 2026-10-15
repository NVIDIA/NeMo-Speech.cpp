import os
import threading
from concurrent.futures import ThreadPoolExecutor


def snapshot(backend, loop=None):
    import torch

    executors = {
        key: value._max_workers
        for key, value in vars(backend).items()
        if isinstance(value, ThreadPoolExecutor)
    }
    default = getattr(loop, "_default_executor", None)
    return dict(
        scope="preparation_only",
        affinity=sorted(os.sched_getaffinity(0)),
        os_cpu_count=os.cpu_count(),
        affinity_cpu_count=len(os.sched_getaffinity(0)),
        torch_intraop=torch.get_num_threads(),
        torch_interop=torch.get_num_interop_threads(),
        backend_executor_max_workers=executors,
        asyncio_default_executor_max_workers=getattr(default, "_max_workers", None),
        environment={
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKIO_WORKER_THREADS",
                "NEMOTRON_R21_TORCH_THREADS",
                "NEMOTRON_R21_TORCH_INTEROP_THREADS",
            )
        },
        python_threads=[
            dict(name=thread.name, native_id=thread.native_id) for thread in threading.enumerate()
        ],
    )


def configure(*, intraop_threads, interop_threads):
    import torch

    for name, count, setter in [
        ("intraop", intraop_threads, torch.set_num_threads),
        ("interop", interop_threads, torch.set_num_interop_threads),
    ]:
        if type(count) is not int or count < 1:
            raise ValueError("Invalid thread-pool size: " + name)
        setter(count)
