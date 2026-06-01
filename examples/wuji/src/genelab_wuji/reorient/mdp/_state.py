"""Lazily-allocated per-env runtime caches for reorient terms.

Manager terms are plain callables, so cross-term state (the cage escape counter, the
joint-velocity history for the acceleration penalty) lives on the env and is created on
first access. Reset events zero the relevant slices.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def _buf(
    env: "EnvContext", attr: str, *shape: int, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    buf = getattr(env, attr, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, *shape, dtype=dtype, device=env.device)
        setattr(env, attr, buf)
    return buf


def cage_counter(env: "EnvContext") -> torch.Tensor:
    """Per-env soft cage-escape counter (float, in steps)."""
    return _buf(env, "_wuji_cage_counter")


def prev_joint_vel(env: "EnvContext", num_dofs: int) -> torch.Tensor:
    """Per-env previous-step joint velocity cache for the acceleration penalty."""
    return _buf(env, "_wuji_prev_joint_vel", num_dofs)
