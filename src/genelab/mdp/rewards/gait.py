"""Locomotion gait-shaping reward terms.

Port of mjlab's body / foot gait-shaping rewards from ``tasks/velocity/mdp/rewards.py``.
Each gates on ``command magnitude > command_threshold`` so the penalty is silent when the
policy is asked to stand still — otherwise the standing envs would pile up free penalty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import (
    asset_state as _asset_state,
    command_active as _command_active,
    contact_sensor as _contact_sensor,
    link_ids as _link_ids,
    site_lin_vel_w as _site_lin_vel_w,
    site_pos_w as _site_pos_w,
)
from genelab.sensor.self_contact import SelfContactSensor

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.managers.reward_manager import RewardTermCfg
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``Σ ω_xy²`` across the links named by ``asset_cfg`` (typical G1 use: torso only).

    mjlab: ``tasks/velocity/mdp/rewards.py::body_angular_velocity_penalty``. With a
    single ``link_names=("torso_link",)`` selector the output matches mjlab's
    single-body variant; multiple links sum their contributions.
    """
    indices = list(_link_ids(asset_cfg))
    ang_vel = _asset_state(env, asset_cfg).link_ang_vel_w[:, indices, :2]
    return torch.sum(ang_vel * ang_vel, dim=(-1, -2))


def feet_clearance(
    env: ManagerBasedRlEnv,
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
    foot_vel_xy = _site_lin_vel_w(env, indices, offsets, asset_cfg)[..., :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (B, F)

    if height_sensor_name is not None:
        heights = env.sensors[height_sensor_name].data  # (B, F)
        if heights.shape[-1] != len(indices):
            raise ValueError(
                f"sensor {height_sensor_name!r} returned {heights.shape[-1]} frames, "
                f"expected {len(indices)} to match asset_cfg link order"
            )
    else:
        heights = _site_pos_w(env, indices, offsets, asset_cfg)[..., 2]

    delta = (heights - target_height).abs()
    cost = torch.sum(delta * vel_norm, dim=-1)
    return cost * _command_active(env, command_name, command_threshold)


def feet_slip(
    env: ManagerBasedRlEnv,
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
    foot_vel_xy = _site_lin_vel_w(env, indices, asset_cfg.link_offsets_tensor, asset_cfg)[..., :2]
    vel_sq = torch.sum(foot_vel_xy * foot_vel_xy, dim=-1)  # (B, F)
    cost = torch.sum(vel_sq * in_contact, dim=-1)
    return cost * _command_active(env, command_name, command_threshold)


def soft_landing(
    env: ManagerBasedRlEnv,
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

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset_cfg = asset_cfg
        indices = list(_link_ids(asset_cfg))
        self._foot_indices: list[int] = indices
        self._foot_indices_tensor = torch.tensor(indices, dtype=torch.long, device=env.device)
        self._offsets_tensor: torch.Tensor | None = asset_cfg.link_offsets_tensor
        self._peak_heights = torch.zeros(env.num_envs, len(indices), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        del asset_cfg  # consumed at __init__
        data = _contact_sensor(env, sensor_name).data
        foot_z = _site_pos_w(env, self._foot_indices, self._offsets_tensor, self._asset_cfg)[..., 2]

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


def angular_momentum_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
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
    env: ManagerBasedRlEnv,
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
    env: ManagerBasedRlEnv,
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
