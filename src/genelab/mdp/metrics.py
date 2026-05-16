"""Reusable diagnostic metric functions for :class:`MetricsManager`.

Unlike rewards, metrics don't drive training — they're per-step scalars routed to
tensorboard via ``env.extras["log"]["Episode_Metrics/<name>"]``. Useful for surface
indicators (action smoothness, sensor health, etc.) where a weight-zero reward
term is the wrong tool because it bakes the metric into the reward bookkeeping.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def mean_action_acc(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-env mean ``|a_t − 2·a_{t-1} + a_{t-2}|`` — discrete action acceleration.

    mjlab parity for ``mean_action_acc``. Lower values indicate smoother policy
    outputs (relative to the previous two control steps). Reads the three-slot
    rolling history maintained by :class:`~genelab.managers.action_manager.ActionManager`.
    Right after an env reset both history slots are zero, so the first two
    post-reset steps report inflated values while the buffer fills — that
    transient washes out within an episode for the typical reporting cadence.
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    a_prev_prev = env.action_manager.prev_prev_action
    accel = a - 2.0 * a_prev + a_prev_prev
    return torch.mean(torch.abs(accel), dim=-1)
