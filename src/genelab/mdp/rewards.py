"""Reusable reward term functions for locomotion tasks."""

import re
from typing import TYPE_CHECKING, cast

import torch

from genelab.managers.reward_manager import RewardTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.commands.motion_command import MotionCommand
from genelab.sensor.contact import ContactSensor
from genelab.sensor.self_contact import SelfContactSensor
from genelab.utils.math import quat_apply_inverse, quat_error_magnitude

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _link_ids(asset_cfg: SceneEntityCfg) -> tuple[int, ...]:
    """Pull resolved link ids off an ``asset_cfg`` or fail loudly if they're missing.

    ``asset_cfg`` is normally resolved by ``managers._base.instantiate_class_term``
    at manager construction; this helper double-checks so a misconfigured term
    raises with a useful message instead of silently passing ``None`` to a
    tensor indexer.
    """
    if asset_cfg.link_ids is None:
        raise ValueError(
            f"SceneEntityCfg(name={asset_cfg.name!r}) has no link_ids — set link_names "
            f"and let the manager resolve it, or set link_ids explicitly"
        )
    return asset_cfg.link_ids


def _contact_sensor(env: "ManagerBasedRlEnv", sensor_name: str) -> ContactSensor:
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a ContactSensor (got {type(sensor).__name__})"
        )
    return sensor


def _command_active(env: "ManagerBasedRlEnv", command_name: str, threshold: float) -> torch.Tensor:
    """Returns ``(B,)`` float mask: 1 where ``||cmd[:3]||_2 > threshold``, else 0."""
    cmd = env.command_manager.get_command(command_name)
    mag = torch.norm(cmd[:, :3], dim=-1)
    return (mag > threshold).float()


def track_linear_velocity_xy_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    """``exp(-||cmd_xy - vel_xy||^2 / std^2)``."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = env.robot_state.root_lin_vel_b[:, :2]
    err = torch.sum((cmd - vel) ** 2, dim=-1)
    return torch.exp(-err / (std**2))


def track_angular_velocity_z_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = env.robot_state.root_ang_vel_b[:, 2]
    err = (cmd - vel) ** 2
    return torch.exp(-err / (std**2))


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)


def joint_acc_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # Approximate joint acceleration as joint_vel change per step (best-effort).
    vel = env.robot_state.joint_vel
    return torch.sum(vel**2, dim=-1) * 0.0  # placeholder — proper accel would need history


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
    """L2 of joint-position excursion past ±π (cheap stand-in for true limit penalty)."""
    excess = (env.robot_state.joint_pos.abs() - 3.14).clamp(min=0.0)
    return torch.sum(excess**2, dim=-1)


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

    Height source: if ``height_sensor_name`` is given, the sensor is expected to be a
    multi-frame :class:`~genelab.sensor.TerrainHeightSensor` returning ``(B, F)`` with
    one clearance per foot (column order must match ``asset_cfg.link_names``). Without
    a sensor the reward falls back to ``link_pos.z`` — correct on flat ground.
    """
    indices = list(_link_ids(asset_cfg))
    foot_vel_xy = env.robot_state.link_lin_vel_w[:, indices, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (B, F)

    if height_sensor_name is not None:
        heights = env.sensors[height_sensor_name].data  # (B, F)
        if heights.shape[-1] != len(indices):
            raise ValueError(
                f"sensor {height_sensor_name!r} returned {heights.shape[-1]} frames, "
                f"expected {len(indices)} to match asset_cfg link order"
            )
    else:
        heights = env.robot_state.link_pos[:, indices, 2]

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
    """
    indices = list(_link_ids(asset_cfg))
    in_contact = _contact_sensor(env, sensor_name).data.found.float()
    foot_vel_xy = env.robot_state.link_lin_vel_w[:, indices, :2]
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

    The peak buffer is automatically refreshed at lift-off (``first_detached``) by
    copying the current foot height in, so each new swing measures from scratch. No env
    reset hook is needed: ``first_contact`` only fires after a prior air phase, and that
    prior air phase always starts with a ``first_detached`` reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
        self._env = env
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        indices = list(_link_ids(asset_cfg))
        self._foot_indices = torch.tensor(indices, dtype=torch.long, device=env.device)
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
        foot_z = env.robot_state.link_pos[:, self._foot_indices, 2]

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
    """``||L||₂`` — magnitude of root-frame angular momentum.

    Reads :class:`~genelab.sensor.RootAngularMomentumSensor`'s ``(B, 3)`` vector and
    returns its Euclidean norm. mjlab parity for ``angular_momentum_penalty`` (weight
    −0.02 in the G1 config). Note: GeneLab's sensor uses the **orbital approximation**
    (omits the per-link spin term ``Σ I·ω``) — see ``sensor/angular_momentum.py`` for
    the rationale.
    """
    angmom = env.sensors[sensor_name].data
    return torch.sqrt(torch.sum(angmom * angmom, dim=-1))


def self_collision_cost(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Count of recent self-contact "hit" frames.

    Reads :class:`~genelab.sensor.SelfContactSensor`. When the sensor was configured
    with ``history_length > 0`` the result counts how many frames in the rolling
    window saw a per-env self-contact force sum exceeding ``force_threshold`` —
    mjlab's ``self_collision_cost`` semantic, used in G1 with a 4-step window to
    catch transient sub-step impacts. Without history (``history_length=0``) the
    result is the single-step bool cast to float.
    """
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, SelfContactSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a SelfContactSensor (got {type(sensor).__name__})"
        )
    data = sensor.data
    if data.force_history is not None:
        return (data.force_history > force_threshold).float().sum(dim=-1)
    return (data.force > force_threshold).float()


def feet_air_time(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.4,
) -> torch.Tensor:
    """Stub: reward proportional to mean foot-link height above ground.

    A faithful air-time reward needs contact sensors; until those are wired into the env we
    approximate with the lowest foot z. ``threshold`` controls the height where reward saturates.
    """
    foot_names = env.cfg.robot.foot_link_names
    if not foot_names or not env.link_names:
        return torch.zeros(env.num_envs, device=env.device)
    indices = [env.link_names.index(n) for n in foot_names if n in env.link_names]
    if not indices:
        return torch.zeros(env.num_envs, device=env.device)
    foot_z = env.robot_state.link_pos[:, indices, 2]
    height = foot_z.mean(dim=-1).clamp(0.0, threshold)
    return height / threshold


# --------------------------------------------------------------------- motion imitation


def _motion_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def _body_index_filter(cmd: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [
        i
        for i, name in enumerate(cmd.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """``exp(-||p_ref - p_robot||^2 / std^2)`` on the anchor body in world frame."""
    cmd = _motion_command(env, command_name)
    error = torch.sum((cmd.anchor_pos_w - cmd.robot_anchor_pos_w) ** 2, dim=-1)
    return torch.exp(-error / (std * std))


def motion_global_anchor_orientation_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """Geodesic rotation error on the anchor body, mapped through a Gaussian kernel."""
    cmd = _motion_command(env, command_name)
    error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-error / (std * std))


def motion_relative_body_position_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body L2 position error against the anchor-aligned reference frames."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_pos_relative_w[:, indexes] - cmd.robot_body_pos_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_relative_body_orientation_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body geodesic rotation error against the anchor-aligned reference frames."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = (
        quat_error_magnitude(
            cmd.body_quat_relative_w[:, indexes],
            cmd.robot_body_quat_w[:, indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_global_body_linear_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame linear-velocity error across the tracked bodies."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_lin_vel_w[:, indexes] - cmd.robot_body_lin_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_global_body_angular_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame angular-velocity error across the tracked bodies."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_ang_vel_w[:, indexes] - cmd.robot_body_ang_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))
