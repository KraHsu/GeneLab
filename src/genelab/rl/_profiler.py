"""Opt-in ``torch.profiler`` wrapper for diagnosing training slowdowns.

Activated by ``GENELAB_PROFILE=1`` in the environment. Disabled by default so existing
runs are unaffected. Only rank 0 emits a trace under distributed launches, to avoid every
worker writing simultaneously.

Environment variables (all optional):

* ``GENELAB_PROFILE=1`` — turn the profiler on. Any other value is treated as off.
* ``GENELAB_PROFILE_OUT`` — directory the TensorBoard trace handler writes to. Default
  ``logs/torch_profile`` (relative to the current working directory).
* ``GENELAB_PROFILE_WAIT`` / ``_WARMUP`` / ``_ACTIVE`` / ``_REPEAT`` — passed straight to
  ``torch.profiler.schedule``. Defaults: ``wait=10``, ``warmup=5``, ``active=10``,
  ``repeat=2``. Tweak if iterations are very long or very short.

The profiler's ``.step()`` must be advanced once per training iteration for the schedule
to move forward. We wire this in by patching the active runner's ``learn`` loop one level
up (see ``genelab.rl.runner.train_task``), which inserts a profiler-step callback. If the
patching path isn't taken, the context manager still records a single contiguous trace.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genelab.rl.distributed import is_main_process

if TYPE_CHECKING:
    from collections.abc import Generator


def profiler_enabled() -> bool:
    return os.environ.get("GENELAB_PROFILE", "0") == "1"


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@contextmanager
def maybe_profile() -> "Generator[Any, None, None]":
    """Yield a profiler-step callback. No-op when ``GENELAB_PROFILE != 1`` or not rank 0."""
    if not profiler_enabled() or not is_main_process():
        yield None
        return
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    out_dir = Path(os.environ.get("GENELAB_PROFILE_OUT", "logs/torch_profile"))
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(
        wait=_read_int("GENELAB_PROFILE_WAIT", 10),
        warmup=_read_int("GENELAB_PROFILE_WARMUP", 5),
        active=_read_int("GENELAB_PROFILE_ACTIVE", 10),
        repeat=_read_int("GENELAB_PROFILE_REPEAT", 2),
    )
    handler = tensorboard_trace_handler(str(out_dir))

    with profile(activities=activities, schedule=sched, on_trace_ready=handler) as prof:
        yield prof.step
