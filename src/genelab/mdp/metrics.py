"""Reusable diagnostic metric functions for :class:`MetricsManager`.

Unlike rewards, metrics don't drive training — they're per-step scalars routed to
tensorboard via ``env.extras["log"]["Episode_Metrics/<name>"]``. Useful for surface
indicators (action smoothness, sensor health, etc.) where a weight-zero reward
term is the wrong tool because it bakes the metric into the reward bookkeeping.

mjlab logs the corresponding diagnostics as ``Metrics/<name>_mean`` directly from
inside the reward functions, then rsl-rl averages them across a rollout. GeneLab's
:class:`~genelab.managers.metrics_manager.MetricsManager` instead averages over
an episode and reports at env reset under ``Episode_Metrics/<name>``. The
per-step scalar is the same; the time horizon (rollout vs episode) differs by
a constant factor for diagnostic comparison.
"""

from typing import TYPE_CHECKING

import torch

from genelab.managers.reward_manager import RewardTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import (
    contact_sensor as _contact_sensor,
    link_ids as _link_ids,
    site_lin_vel_w as _site_lin_vel_w,
    site_pos_w as _site_pos_w,
)

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def mean_action_acc(env: "EnvContext") -> torch.Tensor:
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


# --------------------------------------------------------------------- gait diagnostics
#
# The five metrics below mirror mjlab's ``Metrics/{air_time,peak_height,
# slip_velocity,landing_force,angular_momentum}_mean`` writes (see
# ``mjlab/tasks/velocity/mdp/rewards.py:205, 229, 317, 352, 373``). Each
# computes mjlab's exact cross-(env, foot) scalar each step, then broadcasts
# to ``(B,)`` so :class:`MetricsManager` can accumulate it uniformly across
# envs — the broadcast collapses cleanly at reset because every env sees the
# same value.


def _broadcast_scalar(value: torch.Tensor, env: "EnvContext") -> torch.Tensor:
    """Expand a 0-dim tensor to ``(B,)`` so MetricsManager can accumulate per-env."""
    return value.expand(env.num_envs)


def angular_momentum_mean(env: "EnvContext", sensor_name: str) -> torch.Tensor:
    """``||L||`` of the root-frame angular momentum.

    mjlab logs ``torch.mean(angmom_magnitude)`` (a cross-env scalar) inside the
    ``angular_momentum_penalty`` reward; per-env ``||L||`` averaged over the
    episode gives the same per-batch value when MetricsManager reduces across
    envs at reset.
    """
    angmom = env.sensors[sensor_name].data
    return torch.norm(angmom, dim=-1)


def air_time_mean(env: "EnvContext", sensor_name: str) -> torch.Tensor:
    """Mean ``current_air_time`` across (env, foot) pairs that are in the air.

    Replicates mjlab's ``Metrics/air_time_mean`` in ``feet_air_time``: sum of
    air times divided by number of mid-air feet (clamped to 1). Broadcasted to
    ``(B,)`` for MetricsManager.
    """
    data = _contact_sensor(env, sensor_name).data
    current_air_time = data.current_air_time
    in_air = current_air_time > 0
    in_air_f = in_air.float()
    num_in_air = in_air_f.sum().clamp(min=1.0)
    mean = (current_air_time * in_air_f).sum() / num_in_air
    return _broadcast_scalar(mean, env)


def slip_velocity_mean(
    env: "EnvContext", sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Mean ``||v_xy||`` across grounded (env, foot) pairs.

    Replicates mjlab's ``Metrics/slip_velocity_mean`` in ``feet_slip``. Reads
    site-frame foot velocity (``link_offsets`` honoured) and weights by
    in-contact bool. Broadcasted to ``(B,)``.
    """
    indices = list(_link_ids(asset_cfg))
    data = _contact_sensor(env, sensor_name).data
    in_contact = data.found.float()  # (B, F)
    foot_vel_xy = _site_lin_vel_w(env, indices, asset_cfg.link_offsets_tensor)[..., :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (B, F)
    num_in_contact = in_contact.sum().clamp(min=1.0)
    mean = (vel_norm * in_contact).sum() / num_in_contact
    return _broadcast_scalar(mean, env)


def landing_force_mean(env: "EnvContext", sensor_name: str) -> torch.Tensor:
    """Mean ``|F|`` across (env, foot) pairs that just landed.

    Replicates mjlab's ``Metrics/landing_force_mean`` in ``soft_landing``:
    contact-force magnitude on the substep ``first_contact`` fires, averaged
    over the landings (clamped to 1). Broadcasted to ``(B,)``.
    """
    data = _contact_sensor(env, sensor_name).data
    first_contact = data.first_contact.float()
    landing_impact = data.force_norm * first_contact
    num_landings = first_contact.sum().clamp(min=1.0)
    mean = landing_impact.sum() / num_landings
    return _broadcast_scalar(mean, env)


class peak_height_mean:
    """Mean swing-apex height across (env, foot) pairs at landing.

    Stateful: tracks per-foot peak height during the air phase, samples the
    peak on the substep ``first_contact`` fires, then resets the slot for the
    next swing. Mirrors :class:`~genelab.mdp.rewards.feet_swing_height`'s
    bookkeeping (separate buffer so the metric works whether the reward is
    weight=0 or active — currently weight=−0.25 in G1).

    Replicates mjlab's ``Metrics/peak_height_mean`` in ``feet_swing_height``
    (``mjlab/.../rewards.py:317``). Broadcasted to ``(B,)`` for MetricsManager.
    """

    def __init__(self, cfg: RewardTermCfg, env: "EnvContext") -> None:
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        indices = list(_link_ids(asset_cfg))
        self._foot_indices: list[int] = indices
        self._offsets_tensor: torch.Tensor | None = asset_cfg.link_offsets_tensor
        self._peak_heights = torch.zeros(env.num_envs, len(indices), device=env.device)

    def __call__(
        self,
        env: "EnvContext",
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del asset_cfg  # consumed at __init__
        data = _contact_sensor(env, sensor_name).data
        foot_z = _site_pos_w(env, self._foot_indices, self._offsets_tensor)[..., 2]
        # Accumulate the peak while airborne.
        in_air = ~data.found
        self._peak_heights = torch.where(
            in_air, torch.maximum(self._peak_heights, foot_z), self._peak_heights
        )
        first_contact = data.first_contact.float()
        peak_at_landing = self._peak_heights * first_contact
        num_landings = first_contact.sum().clamp(min=1.0)
        mean = peak_at_landing.sum() / num_landings
        # Zero the peak buffer for feet that just landed so the next swing
        # starts fresh.
        self._peak_heights = torch.where(
            data.first_contact, torch.zeros_like(self._peak_heights), self._peak_heights
        )
        return _broadcast_scalar(mean, env)
