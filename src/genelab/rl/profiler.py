"""Opt-in ``torch.profiler`` wrapper for diagnosing GeneLab perf.

Disabled by default so existing runs are unaffected. Each knob accepts both an explicit
keyword argument (passed by the CLI / library callers) and an environment variable
fallback. The kwarg wins when both are set; an unset kwarg falls back to the env var; an
unset env var falls back to the built-in default. Only rank 0 emits a trace under
distributed launches, so workers do not write simultaneously.

Environment variables (all optional, all matched by a ``maybe_profile`` kwarg):

* ``GENELAB_PROFILE=1`` — turn the profiler on (``enabled=True``). Any other value is off.
* ``GENELAB_PROFILE_OUT`` — directory the TensorBoard trace handler writes to
  (``out_dir``). Default ``logs/torch_profile`` (relative to the current working dir).
* ``GENELAB_PROFILE_WAIT`` / ``_WARMUP`` / ``_ACTIVE`` / ``_REPEAT`` — passed straight to
  ``torch.profiler.schedule`` (``wait``, ``warmup``, ``active``, ``repeat``). Defaults:
  ``wait=10``, ``warmup=5``, ``active=10``, ``repeat=2``.
* ``GENELAB_PROFILE_RECORD_SHAPES=1`` — record tensor shapes (``record_shapes``).
* ``GENELAB_PROFILE_WITH_STACK=1`` — capture Python stack traces (``with_stack``). Useful
  for source-attribution in TensorBoard, but adds noticeable overhead.

The profiler's ``.step()`` must be advanced once per iteration for the schedule to move
forward. ``genelab.rl.runner`` wires this into both ``train_task`` (via a step hook on
``RslRlVecEnvWrapper.step``) and ``play_task`` (inside the rollout loop). If the wiring
is bypassed, the context manager still records a single contiguous trace.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genelab.utils.distributed import is_main_process

if TYPE_CHECKING:
    from collections.abc import Generator


_DEFAULT_OUT_DIR = "logs/torch_profile"
_DEFAULT_WAIT = 10
_DEFAULT_WARMUP = 5
_DEFAULT_ACTIVE = 10
_DEFAULT_REPEAT = 2


def profiler_enabled() -> bool:
    """Return True when ``GENELAB_PROFILE=1``. Does not consult kwargs."""
    return os.environ.get("GENELAB_PROFILE", "0") == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _resolve_enabled(enabled: bool | None) -> bool:
    return profiler_enabled() if enabled is None else enabled


def _resolve_out_dir(out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    return Path(os.environ.get("GENELAB_PROFILE_OUT", _DEFAULT_OUT_DIR))


def _resolve_int(kwarg: int | None, env_name: str, default: int) -> int:
    return _env_int(env_name, default) if kwarg is None else kwarg


def _resolve_bool(kwarg: bool | None, env_name: str) -> bool:
    return _env_bool(env_name) if kwarg is None else kwarg


@contextmanager
def maybe_profile(
    *,
    enabled: bool | None = None,
    out_dir: Path | None = None,
    wait: int | None = None,
    warmup: int | None = None,
    active: int | None = None,
    repeat: int | None = None,
    record_shapes: bool | None = None,
    with_stack: bool | None = None,
) -> "Generator[Any, None, None]":
    """Yield a profiler-step callback. No-op when disabled or off the main rank.

    Each keyword argument overrides the matching ``GENELAB_PROFILE_*`` env var when set.
    """
    if not _resolve_enabled(enabled) or not is_main_process():
        yield None
        return
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    resolved_out = _resolve_out_dir(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(
        wait=_resolve_int(wait, "GENELAB_PROFILE_WAIT", _DEFAULT_WAIT),
        warmup=_resolve_int(warmup, "GENELAB_PROFILE_WARMUP", _DEFAULT_WARMUP),
        active=_resolve_int(active, "GENELAB_PROFILE_ACTIVE", _DEFAULT_ACTIVE),
        repeat=_resolve_int(repeat, "GENELAB_PROFILE_REPEAT", _DEFAULT_REPEAT),
    )
    handler = tensorboard_trace_handler(str(resolved_out))

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=_resolve_bool(record_shapes, "GENELAB_PROFILE_RECORD_SHAPES"),
        with_stack=_resolve_bool(with_stack, "GENELAB_PROFILE_WITH_STACK"),
    ) as prof:
        yield prof.step
