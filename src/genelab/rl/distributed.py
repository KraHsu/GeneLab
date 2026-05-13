"""Helpers for torchrun-managed distributed training.

The training pipeline relies on three env vars set by ``torchrun``:

* ``WORLD_SIZE`` - total number of ranks across all nodes
* ``LOCAL_RANK`` - rank index on the current node (used to pick the GPU)
* ``RANK``       - global rank index (used to gate logging on rank 0)

When ``WORLD_SIZE == 1`` (the default), these helpers no-op so single-GPU
training paths stay untouched.
"""

import os
from typing import Final

_RANK_ENV: Final[str] = "RANK"
_LOCAL_RANK_ENV: Final[str] = "LOCAL_RANK"
_WORLD_SIZE_ENV: Final[str] = "WORLD_SIZE"


def world_size() -> int:
    return int(os.environ.get(_WORLD_SIZE_ENV, "1"))


def local_rank() -> int:
    return int(os.environ.get(_LOCAL_RANK_ENV, "0"))


def global_rank() -> int:
    return int(os.environ.get(_RANK_ENV, "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    return global_rank() == 0


def pin_cuda_device() -> str | None:
    """Return the canonical ``cuda:{LOCAL_RANK}`` string for this worker.

    The bootstrap in ``genelab.cli._bootstrap`` has already rewritten
    ``LOCAL_RANK`` to ``0`` and pinned ``CUDA_VISIBLE_DEVICES`` to a single
    physical GPU (Quadrants, Genesis's compute backend, only honors
    ``CUDA_VISIBLE_DEVICES``, not ``torch.cuda.set_device()``). So in every
    distributed worker this returns ``"cuda:0"`` — the rank's only visible
    device — which also matches what rsl_rl's ``OnPolicyRunner`` expects given
    the rewritten ``LOCAL_RANK``.

    The explicit ``torch.cuda.set_device`` call is kept defensively: if the
    bootstrap was bypassed, it still aligns ``torch.cuda.current_device()``
    so ``gs.init`` picks the visible device.
    """
    if not is_distributed():
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requested (WORLD_SIZE>1) but torch.cuda is unavailable"
        )
    rank = local_rank()
    torch.cuda.set_device(rank)
    return f"cuda:{rank}"
