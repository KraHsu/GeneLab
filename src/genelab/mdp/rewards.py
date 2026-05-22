"""Reusable reward term functions for locomotion tasks."""

import re
import warnings
from typing import TYPE_CHECKING

import torch

from genelab.managers.reward_manager import RewardTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import (
    command_active as _command_active,
    contact_sensor as _contact_sensor,
    link_ids as _link_ids,
    site_lin_vel_w as _site_lin_vel_w,
    site_pos_w as _site_pos_w,
)
from genelab.mdp.motion_tracking import (
    motion_global_anchor_orientation_error_exp as motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp as motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp as motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp as motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp as motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp as motion_relative_body_position_error_exp,
)
from genelab.sensor.self_contact import SelfContactSensor
from genelab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def track_linear_velocity_xy_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    """``exp(-(||cmd_xy - vel_xy||² + vel_z²) / std²)``.

    mjlab parity: assumes the commanded z-velocity is zero, so any non-zero
    vertical motion contributes to the tracking error. Discourages vertical
    bouncing alongside xy-tracking.
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = env.robot_state.root_lin_vel_b
    xy_err = torch.sum((cmd - vel[:, :2]) ** 2, dim=-1)
    z_err = vel[:, 2] ** 2
    return torch.exp(-(xy_err + z_err) / (std**2))


def track_angular_velocity_z_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    """``exp(-((cmd_z − vel_z)² + ||vel_xy||²) / std²)``.

    mjlab parity: assumes the commanded xy angular velocities are zero, so any
    pitching/rolling rate contributes to the error term. Discourages tumbling
    alongside yaw-tracking.
    """
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = env.robot_state.root_ang_vel_b
    z_err = (cmd - vel[:, 2]) ** 2
    xy_err = torch.sum(vel[:, :2] ** 2, dim=-1)
    return torch.exp(-(z_err + xy_err) / (std**2))


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)


# --------------------------------------------------------------------- base hard-constraints (M2.4)


def lin_vel_z_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize vertical base velocity — ``v_z²`` in the base frame.

    Discourages bouncing / vertical oscillation in locomotion. Returns the
    non-negative square; pair it with a negative weight in the term cfg.
    """
    return env.robot_state.root_lin_vel_b[:, 2] ** 2


def base_height_l2(env: "ManagerBasedRlEnv", target_height: float) -> torch.Tensor:
    """Penalize squared deviation of base height from ``target_height``.

    Flat-ground variant (no terrain-height sensor): ``(root_z − target_height)²``
    on the world-frame root z. Pair with a negative weight.
    """
    return (env.robot_state.root_pos[:, 2] - target_height) ** 2


def alive_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Constant ``+1`` per env per step (pair with a positive weight).

    Rewards staying alive so per-step penalties don't make early termination look
    attractive. The reward manager zeroes terminated envs on reset, so this counts
    only steps actually taken.
    """
    return torch.ones(env.num_envs, device=env.device)


def applied_torque_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize squared realized actuator torque — ``Σⱼ τⱼ²`` over actuated joints.

    Reads ``robot_state.applied_torque`` (Genesis control force, refreshed each step).
    Discourages high-effort policies; pair with a negative weight.
    """
    return torch.sum(env.robot_state.applied_torque**2, dim=-1)


def joint_vel_limits(env: "ManagerBasedRlEnv", soft_ratio: float = 1.0) -> torch.Tensor:
    """Sum of per-joint speed excursions past ``soft_ratio × joint_vel_limit``.

    Mirrors :func:`joint_pos_limits` for velocity: ``Σⱼ max(0, |q̇ⱼ| − ratio·limⱼ)``.
    The limit comes from ``env.joint_vel_limits`` (``ArticulationCfg.joint_vel_limit``);
    joints with a ``+∞`` limit contribute zero, so this is inert until a task opts in.
    """
    limit = env.joint_vel_limits * soft_ratio  # (J,)
    speed = torch.abs(env.robot_state.joint_vel)  # (B, J)
    return torch.sum((speed - limit.unsqueeze(0)).clamp(min=0.0), dim=-1)


_joint_acc_l2_warned = False


def joint_acc_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
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
    return torch.zeros(env.robot_state.joint_vel.shape[0], device=env.robot_state.joint_vel.device)


def flat_orientation_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalise tilt: the xy components of body-frame gravity should be zero."""
    return torch.sum(env.robot_state.projected_gravity_b[:, :2] ** 2, dim=-1)


def upright_exp(
    env: "ManagerBasedRlEnv",
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
        xy_squared = torch.sum(env.robot_state.projected_gravity_b[:, :2] ** 2, dim=-1)
        return torch.exp(-xy_squared / (std * std))
    # ``gravity_w = (0, 0, -1)`` is the convention used throughout robot_state — project
    # it into each selected link's frame via the link's world quaternion. Result xy
    # components measure tilt of that link about the gravity vector.
    link_ids = list(asset_cfg.link_ids)
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_w[:, 2] = -1.0
    quat_w = env.robot_state.link_quat_w[:, link_ids, :]  # (B, N, 4)
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

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
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
        out = torch.full((len(self._env.joint_names),), default, device=self._env.device)
        for pattern, value in mapping.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))
            for i, name in enumerate(self._env.joint_names):
                if regex.fullmatch(name) or regex.search(name):
                    out[i] = float(value)
        return out

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
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

        error = env.robot_state.joint_pos - env.default_joint_pos
        return torch.exp(-torch.mean((error * error) / (std * std), dim=-1))


def joint_pos_limits(env: "ManagerBasedRlEnv") -> torch.Tensor:
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
    limits = env.joint_pos_limits  # (J, 2)
    lower = limits[:, 0]
    upper = limits[:, 1]
    joint_pos = env.robot_state.joint_pos  # (B, J)
    out_of_lower = (lower.unsqueeze(0) - joint_pos).clamp(min=0.0)
    out_of_upper = (joint_pos - upper.unsqueeze(0)).clamp(min=0.0)
    return torch.sum(out_of_lower + out_of_upper, dim=-1)


# --------------------------------------------------------------------- locomotion gait shaping
# Port of mjlab's body / foot gait-shaping rewards from ``tasks/velocity/mdp/rewards.py``.
# Each gates on ``command magnitude > command_threshold`` so the penalty is silent when the
# policy is asked to stand still — otherwise the standing envs would pile up free penalty.


def body_angular_velocity_penalty(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``Σ ω_xy²`` across the links named by ``asset_cfg`` (typical G1 use: torso only).

    mjlab: ``tasks/velocity/mdp/rewards.py::body_angular_velocity_penalty``. With a
    single ``link_names=("torso_link",)`` selector the output matches mjlab's
    single-body variant; multiple links sum their contributions.
    """
    indices = list(_link_ids(asset_cfg))
    ang_vel = env.robot_state.link_ang_vel_w[:, indices, :2]
    return torch.sum(ang_vel * ang_vel, dim=(-1, -2))


def feet_clearance(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str,
    command_threshold: float = 0.05,
    height_sensor_name: str | None = None,
) -> torch.Tensor:
    """``Σ_foot |h − target| · |v_xy|`` while command is active.

    Penalises foot height deviation from ``target_height`` weighted by horizontal foot
    velocity — so feet are pushed toward the target swing height only while they're
    actually moving. mjlab: ``feet_clearance``.

    Honors ``asset_cfg.link_offsets`` (mjlab site parity): when set, the foot
    velocity used here is ``v_link + ω × (R · offset)`` and, when no height
    sensor is given, the fallback height uses the site-frame z. The
    ``height_sensor_name`` path delegates to a multi-frame
    :class:`~genelab.sensor.TerrainHeightSensor`, which applies its own
    ``link_offsets`` to the ray origin.
    """
    indices = list(_link_ids(asset_cfg))
    offsets = asset_cfg.link_offsets_tensor
    foot_vel_xy = _site_lin_vel_w(env, indices, offsets)[..., :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (B, F)

    if height_sensor_name is not None:
        heights = env.sensors[height_sensor_name].data  # (B, F)
        if heights.shape[-1] != len(indices):
            raise ValueError(
                f"sensor {height_sensor_name!r} returned {heights.shape[-1]} frames, "
                f"expected {len(indices)} to match asset_cfg link order"
            )
    else:
        heights = _site_pos_w(env, indices, offsets)[..., 2]

    delta = (heights - target_height).abs()
    cost = torch.sum(delta * vel_norm, dim=-1)
    return cost * _command_active(env, command_name, command_threshold)


def feet_slip(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """``Σ_foot ||v_xy||² · in_contact`` — penalise horizontal foot slip while grounded.

    mjlab: ``feet_slip``. Gated by command magnitude so standing envs don't accumulate.
    Honors ``asset_cfg.link_offsets`` for site-frame velocity (mjlab parity).
    """
    indices = list(_link_ids(asset_cfg))
    in_contact = _contact_sensor(env, sensor_name).data.found.float()
    foot_vel_xy = _site_lin_vel_w(env, indices, asset_cfg.link_offsets_tensor)[..., :2]
    vel_sq = torch.sum(foot_vel_xy * foot_vel_xy, dim=-1)  # (B, F)
    cost = torch.sum(vel_sq * in_contact, dim=-1)
    return cost * _command_active(env, command_name, command_threshold)


def soft_landing(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """``Σ_foot |F| · first_contact`` — penalise contact force spikes at touchdown.

    mjlab: ``soft_landing``. Reads ``ContactData.first_contact`` (added in this PR) so the
    impulse is only charged on the step a foot transitions air→contact. Requires
    ``track_air_time=True`` on the sensor.
    """
    data = _contact_sensor(env, sensor_name).data
    landing = data.first_contact.float()
    cost = torch.sum(data.force_norm * landing, dim=-1)
    return cost * _command_active(env, command_name, command_threshold)


class feet_swing_height:
    """``Σ_foot (peak_h / target − 1)² · first_contact`` evaluated at each landing.

    Tracks per-foot peak height during the current swing phase, then at the moment of
    foot touchdown emits a cost proportional to how far the swing apex was from
    ``target_height``. mjlab: ``feet_swing_height``.

    Honors ``asset_cfg.link_offsets`` (mjlab site parity): the tracked height is the
    site-frame z, ``link_z + (R_link · offset)_z``, so a foot site sitting 0.037 m
    below the ankle_roll_link origin measures swing height correctly.

    The peak buffer is automatically refreshed at lift-off (``first_detached``) by
    copying the current foot height in, so each new swing measures from scratch. No env
    reset hook is needed: ``first_contact`` only fires after a prior air phase, and that
    prior air phase always starts with a ``first_detached`` reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
        self._env = env
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        indices = list(_link_ids(asset_cfg))
        self._foot_indices: list[int] = indices
        self._foot_indices_tensor = torch.tensor(indices, dtype=torch.long, device=env.device)
        self._offsets_tensor: torch.Tensor | None = asset_cfg.link_offsets_tensor
        self._peak_heights = torch.zeros(env.num_envs, len(indices), device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        del asset_cfg  # consumed at __init__
        data = _contact_sensor(env, sensor_name).data
        foot_z = _site_pos_w(env, self._foot_indices, self._offsets_tensor)[..., 2]

        # On lift-off, snap the peak to the current height so the new swing measures fresh.
        self._peak_heights = torch.where(data.first_detached, foot_z, self._peak_heights)
        # While airborne, accumulate the peak.
        in_air = ~data.found
        self._peak_heights = torch.where(
            in_air, torch.maximum(self._peak_heights, foot_z), self._peak_heights
        )

        error = self._peak_heights / target_height - 1.0
        landing = data.first_contact.float()
        cost = torch.sum(error.pow(2) * landing, dim=-1)
        return cost * _command_active(env, command_name, command_threshold)


def angular_momentum_penalty(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """``||L||₂²`` — squared magnitude of root-frame angular momentum.

    Reads :class:`~genelab.sensor.RootAngularMomentumSensor`'s ``(B, 3)`` vector
    and returns its squared Euclidean norm. mjlab parity — see
    ``mjlab/tasks/velocity/mdp/rewards.py::angular_momentum_penalty``, which
    returns ``angmom_magnitude_sq`` (i.e. squared norm). The quadratic curve
    penalises large momenta much harder than small ones; with weight −0.02 the
    G1 reference uses this shape to discourage flailing.

    GeneLab's underlying sensor uses the **orbital approximation** (omits the
    per-link spin term ``Σ I·ω``) — see ``sensor/angular_momentum.py``.
    """
    angmom = env.sensors[sensor_name].data
    return torch.sum(angmom * angmom, dim=-1)


def self_collision_cost(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Count of recent self-contact "hit" frames.

    Reads :class:`~genelab.sensor.SelfContactSensor`. When the sensor was configured
    with ``history_length > 0`` the result counts how many substeps in the rolling
    window saw at least one self-contact pair above the sensor's ``force_threshold``
    — mjlab's ``self_collision_cost`` semantic (``force_history.any(dim=pair_axis).sum``),
    used in G1 with a 4-step window. Without history (``history_length=0``) the
    result is the single-step bool cast to float.

    The threshold lives on :class:`~genelab.sensor.SelfContactSensorCfg.force_threshold`
    (not here). It has to: the sensor compresses to a bool *before* history
    accumulation because Genesis contact-pair indices reshuffle each step, and
    deferring the threshold to the reward would lose the per-pair breakdown.
    A pre-parity ``force_threshold`` reward parameter was dropped here in favour
    of the single source of truth on the sensor cfg.
    """
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, SelfContactSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a SelfContactSensor (got {type(sensor).__name__})"
        )
    data = sensor.data
    if data.force_history is not None:
        return data.force_history.float().sum(dim=-1)
    return data.found.float()


def feet_air_time(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Count of feet whose current air time is in ``(threshold_min, threshold_max)``.

    mjlab parity (``feet_air_time``): reads ``ContactSensor.current_air_time`` and
    counts how many feet are mid-swing within the configured window — encourages
    stepping cadence without rewarding pathologically long swings.

    When ``command_name`` is given, the reward is masked to zero on envs whose
    commanded velocity magnitude is below ``command_threshold`` so the policy
    doesn't get a free signal while standing.

    Note: GeneLab's pre-P5 stub used a foot-z height proxy. This implementation
    now matches mjlab; the G1 reference cfg sets weight=0 so the term doesn't
    fire during training there.
    """
    data = _contact_sensor(env, sensor_name).data
    current_air_time = data.current_air_time
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=-1)
    if command_name is not None:
        reward = reward * _command_active(env, command_name, command_threshold)
    return reward
