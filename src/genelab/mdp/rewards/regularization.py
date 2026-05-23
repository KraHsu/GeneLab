"""Regularization and base hard-constraint reward terms (M2.4).

Effort/rate penalties, base pose/orientation shaping, joint-limit excursions, and
the speed-dependent posture reward — terms that keep the policy well-behaved
independent of the gait pattern.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import (
    asset_articulation as _asset_articulation,
    asset_state as _asset_state,
)
from genelab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
    from genelab.managers.reward_manager import RewardTermCfg
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def action_rate_l2(env: EnvContext) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)


def lin_vel_z_l2(env: EnvContext, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize vertical base velocity — ``v_z²`` in the base frame.

    Discourages bouncing / vertical oscillation in locomotion. Returns the
    non-negative square; pair it with a negative weight in the term cfg.
    """
    return _asset_state(env, asset_cfg).root_lin_vel_b[:, 2] ** 2


def base_height_l2(
    env: EnvContext, target_height: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize squared deviation of base height from ``target_height``.

    Flat-ground variant (no terrain-height sensor): ``(root_z − target_height)²``
    on the world-frame root z. Pair with a negative weight.
    """
    return (_asset_state(env, asset_cfg).root_pos[:, 2] - target_height) ** 2


def alive_bonus(env: EnvContext) -> torch.Tensor:
    """Constant ``+1`` per env per step (pair with a positive weight).

    Rewards staying alive so per-step penalties don't make early termination look
    attractive. The reward manager zeroes terminated envs on reset, so this counts
    only steps actually taken.
    """
    return torch.ones(env.num_envs, device=env.device)


def applied_torque_l2(
    env: EnvContext, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize squared realized actuator torque — ``Σⱼ τⱼ²`` over actuated joints.

    Reads ``robot_state.applied_torque`` (Genesis control force, refreshed each step).
    Discourages high-effort policies; pair with a negative weight.
    """
    return torch.sum(_asset_state(env, asset_cfg).applied_torque ** 2, dim=-1)


def joint_vel_limits(
    env: EnvContext, soft_ratio: float = 1.0, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Sum of per-joint speed excursions past ``soft_ratio × joint_vel_limit``.

    Mirrors :func:`joint_pos_limits` for velocity: ``Σⱼ max(0, |q̇ⱼ| − ratio·limⱼ)``.
    The limit comes from the entity's ``joint_vel_limits`` (``ArticulationCfg.joint_vel_limit``);
    joints with a ``+∞`` limit contribute zero, so this is inert until a task opts in.
    """
    limit = _asset_articulation(env, asset_cfg).joint_vel_limits * soft_ratio  # (J,)
    speed = torch.abs(_asset_state(env, asset_cfg).joint_vel)  # (B, J)
    return torch.sum((speed - limit.unsqueeze(0)).clamp(min=0.0), dim=-1)


_joint_acc_l2_warned = False


def joint_acc_l2(env: EnvContext) -> torch.Tensor:
    # Stub: proper joint acceleration needs a prev-step joint_vel buffer; not yet wired.
    # Returns zero so it's safe to include in reward weights without affecting gradients.
    global _joint_acc_l2_warned
    if not _joint_acc_l2_warned:
        warnings.warn(
            "joint_acc_l2 is a stub: returns 0. A real implementation needs prior-step "
            "joint_vel history (tracked under ROADMAP M2 actuator/DR work).",
            UserWarning,
            stacklevel=2,
        )
        _joint_acc_l2_warned = True
    rs = _asset_state(env, None)
    return torch.zeros(rs.joint_vel.shape[0], device=rs.joint_vel.device)


def flat_orientation_l2(
    env: EnvContext, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalise tilt: the xy components of body-frame gravity should be zero."""
    return torch.sum(_asset_state(env, asset_cfg).projected_gravity_b[:, :2] ** 2, dim=-1)


def upright_exp(
    env: EnvContext,
    std: float = 0.45,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """``exp(-||projected_gravity_xy||^2 / std^2)`` — positive reward for an upright link.

    Port of mjlab's ``upright`` reward (flat-ground variant). Saturates near zero tilt
    instead of growing unbounded like an L2 penalty, which matches the policy gradients
    the reference implementation relies on.

    Body selection:

    * ``asset_cfg=None`` — read the cached root-frame gravity projection
      (``robot_state.projected_gravity_b``). Penalises **pelvis** tilt for a
      floating-base humanoid. Backward-compatible default.
    * ``asset_cfg=SceneEntityCfg(link_names=(L,))`` — project the world gravity
      vector into ``L``'s frame via ``link_quat_w``. mjlab's G1 cfg targets
      ``torso_link`` so the reward penalises **torso** tilt rather than pelvis
      — different signal when the waist joints flex.

    When multiple links are selected, their xy-squared tilts are summed.
    """
    if asset_cfg is None or asset_cfg.link_ids is None:
        xy_squared = torch.sum(_asset_state(env, asset_cfg).projected_gravity_b[:, :2] ** 2, dim=-1)
        return torch.exp(-xy_squared / (std * std))
    # ``gravity_w = (0, 0, -1)`` is the convention used throughout robot_state — project
    # it into each selected link's frame via the link's world quaternion. Result xy
    # components measure tilt of that link about the gravity vector.
    link_ids = list(asset_cfg.link_ids)
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_w[:, 2] = -1.0
    quat_w = _asset_state(env, asset_cfg).link_quat_w[:, link_ids, :]  # (B, N, 4)
    gravity_expanded = gravity_w.unsqueeze(1).expand_as(quat_w[..., :3])  # (B, N, 3)
    projected = quat_apply_inverse(quat_w, gravity_expanded)  # (B, N, 3)
    xy_squared = torch.sum(projected[..., :2] ** 2, dim=(-1, -2))
    return torch.exp(-xy_squared / (std * std))


class variable_posture:
    """Reward for staying near the default pose, with per-joint, speed-dependent tolerance.

    Port of ``mjlab.tasks.velocity.mdp.rewards.variable_posture``. Std dicts map joint name
    regex → std value; per joint the *last* matching pattern wins (same convention as
    ``RobotEntityCfg.default_joint_pos``). Joints with no match keep the supplied ``default``.

    At each step the active std vector is chosen per env from the command magnitude:

    * ``total_speed = ||lin_vel_xy|| + |ang_vel_z|``
    * standing if ``total_speed < walking_threshold``
    * walking if ``walking_threshold <= total_speed < running_threshold``
    * running otherwise

    Reward: ``exp(-mean((joint_pos - default)^2 / std^2))``.
    """

    def __init__(self, cfg: RewardTermCfg, env: EnvContext) -> None:
        self._env = env
        params = cfg.params
        default = float(params.get("default_std", 1.0))
        std_standing = params.get("std_standing", {})
        std_walking = params.get("std_walking", {})
        std_running = params.get("std_running", {})
        self._std_standing = self._build_std(std_standing, default)
        self._std_walking = self._build_std(std_walking, default)
        self._std_running = self._build_std(std_running, default)

    def _build_std(self, mapping: dict[str, float], default: float) -> torch.Tensor:
        joint_names = _asset_articulation(self._env, None).joint_names
        out = torch.full((len(joint_names),), default, device=self._env.device)
        for pattern, value in mapping.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))
            for i, name in enumerate(joint_names):
                if regex.fullmatch(name) or regex.search(name):
                    out[i] = float(value)
        return out

    def __call__(
        self,
        env: EnvContext,
        command_name: str,
        std_standing: dict[str, float] | None = None,
        std_walking: dict[str, float] | None = None,
        std_running: dict[str, float] | None = None,
        default_std: float = 1.0,
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running, default_std  # consumed at __init__

        command = env.command_manager.get_command(command_name)
        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float().unsqueeze(-1)
        walking_mask = (
            ((total_speed >= walking_threshold) & (total_speed < running_threshold))
            .float()
            .unsqueeze(-1)
        )
        running_mask = (total_speed >= running_threshold).float().unsqueeze(-1)

        std = (
            self._std_standing * standing_mask
            + self._std_walking * walking_mask
            + self._std_running * running_mask
        )

        art = _asset_articulation(env, None)
        error = _asset_state(env, None).joint_pos - art.default_joint_pos
        return torch.exp(-torch.mean((error * error) / (std * std), dim=-1))


def joint_pos_limits(
    env: EnvContext, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Sum of per-joint excursions past the actuator's configured limits.

    mjlab parity (``envs/mdp/rewards.py::joint_pos_limits``):

    * Reads each joint's ``(lower, upper)`` limit from Genesis (sliced to the
      actuated DoFs at bind time — see :attr:`Articulation.joint_pos_limits`).
    * Returns ``Σⱼ (max(0, lower − q) + max(0, q − upper))`` per env.
    * **Absolute** excursion (not squared) — matches mjlab's formula. A joint
      sitting exactly at its limit contributes zero; beyond, the penalty grows
      linearly.

    Joints whose limits are ±∞ (e.g. continuous joints) contribute zero from
    the clamps. Joints not configured for the policy (floating-base DoFs)
    were already filtered out of ``joint_pos``.
    """
    limits = _asset_articulation(env, asset_cfg).joint_pos_limits  # (J, 2)
    lower = limits[:, 0]
    upper = limits[:, 1]
    joint_pos = _asset_state(env, asset_cfg).joint_pos  # (B, J)
    out_of_lower = (lower.unsqueeze(0) - joint_pos).clamp(min=0.0)
    out_of_upper = (joint_pos - upper.unsqueeze(0)).clamp(min=0.0)
    return torch.sum(out_of_lower + out_of_upper, dim=-1)
