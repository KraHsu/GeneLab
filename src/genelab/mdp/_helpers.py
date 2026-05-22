"""Shared helpers used across :mod:`genelab.mdp` modules.

These were previously defined in :mod:`genelab.mdp.rewards` with a leading
underscore (module-private). They're now used by :mod:`genelab.mdp.metrics`
too — moving them here gives both modules a public-within-package import path
and lets pyright stop warning about cross-module private access. They are not
re-exported from :mod:`genelab.mdp` (no entry in ``mdp/__init__.py:__all__``);
external code should keep importing the rewards / metrics functions, not these.
"""

from typing import TYPE_CHECKING, Any, cast

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.sensor.contact import ContactSensor
from genelab.utils.math import quat_apply

if TYPE_CHECKING:
    from genelab.entity import Articulation, RobotState
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def resolve_articulation(env: "ManagerBasedRlEnv", name: str) -> "Articulation":
    """Return the articulation named ``name``, else the primary ``env.articulation``.

    Action / command terms route by ``cfg.asset_name`` (ROADMAP M3.6 / ADR-0012 S2): the
    named entity when ``env.articulations`` exists and holds ``name``, otherwise the singular
    primary — so single-robot terms and fake-env tests (which expose only ``env.articulation``)
    are unchanged.
    """
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name]
    # Fallbacks: the singular primary ``env.articulation`` (real env / most fakes), then
    # ``env`` itself for minimal fakes that expose ``joint_names`` / ``default_joint_pos``
    # at env level without a full articulation (duck-typed — hence the cast).
    return cast("Articulation", getattr(env, "articulation", env))


def resolve_robot_state(env: "ManagerBasedRlEnv", name: str) -> "RobotState":
    """Return the named entity's ``RobotState``, falling back to ``env.robot_state``.

    The read-only counterpart to :func:`resolve_articulation` for terms that only consume
    state (e.g. command terms). Falls back to the primary ``env.robot_state`` so fake envs
    that expose state directly — without a full articulation — keep working.
    """
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].data
    return env.robot_state


def asset_state(env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None") -> "RobotState":
    """``RobotState`` for ``asset_cfg``'s entity (M3.6 / ADR-0012 S3); ``None`` → primary.

    Term functions accept an optional ``asset_cfg`` and read state through this so a
    multi-robot task can point any reward / observation / termination at a specific
    articulation by name. ``None`` (the single-robot default) resolves to the primary
    ``"robot"``.
    """
    return resolve_robot_state(env, asset_cfg.name if asset_cfg is not None else "robot")


def asset_articulation(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None"
) -> "Articulation":
    """:class:`Articulation` for ``asset_cfg``'s entity; ``None`` → primary ``"robot"``."""
    return resolve_articulation(env, asset_cfg.name if asset_cfg is not None else "robot")


def asset_handle(env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None") -> Any:
    """Raw Genesis handle for ``asset_cfg``'s entity (for ``set_pos`` / ``set_dofs_*`` etc.).

    ``env.articulations[name].gs_handle`` when present, else the primary ``env.robot`` — so
    DR / event terms that write through the handle stay single-robot-correct and fake-env
    tests (which expose ``env.robot`` directly) are unchanged. ``None`` → primary ``"robot"``.
    """
    name = asset_cfg.name if asset_cfg is not None else "robot"
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].gs_handle
    return env.robot


def link_ids(asset_cfg: SceneEntityCfg) -> tuple[int, ...]:
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


def contact_sensor(env: "ManagerBasedRlEnv", sensor_name: str) -> ContactSensor:
    """Look up a :class:`ContactSensor` by name; raise ``TypeError`` if the wrong type."""
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a ContactSensor (got {type(sensor).__name__})"
        )
    return sensor


def command_active(env: "ManagerBasedRlEnv", command_name: str, threshold: float) -> torch.Tensor:
    """Returns ``(B,)`` float mask: 1 where ``||cmd_xy|| + |cmd_z| > threshold``, else 0.

    mjlab parity: each gait-shaping reward (``feet_clearance`` / ``feet_slip`` /
    ``soft_landing`` / ``feet_swing_height``) computes the gate this way — sum of
    the linear-xy norm and the absolute yaw rate — rather than an L2 norm of the
    full 3-vector. The two formulas diverge near the threshold: for cmd
    ``(0.03, 0, 0.03)`` and threshold 0.05, mjlab's gate fires (0.06 > 0.05)
    while the L2-of-3 version stays silent (≈0.042). Matters whenever the policy
    is asked to do gentle motion that doesn't break the L2 ball but does exceed
    the L1-y-axes envelope.
    """
    cmd = env.command_manager.get_command(command_name)
    total = torch.norm(cmd[:, :2], dim=-1) + torch.abs(cmd[:, 2])
    return (total > threshold).float()


def site_pos_w(
    env: "ManagerBasedRlEnv",
    indices: list[int],
    offsets: torch.Tensor | None,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """World-frame site position for each selected link.

    ``site_pos_w = link_pos_w + R_link · offset_local``. With ``offsets=None``
    this reduces to ``link_pos_w`` — matches the pre-parity behaviour. ``indices`` index
    ``asset_cfg``'s articulation (``None`` → primary).

    Returns ``(B, F, 3)`` aligned with ``indices``.
    """
    rs = asset_state(env, asset_cfg)
    link_pos = rs.link_pos[:, indices]  # (B, F, 3)
    if offsets is None:
        return link_pos
    link_quat = rs.link_quat_w[:, indices]  # (B, F, 4)
    offsets_b = offsets.unsqueeze(0).expand_as(link_pos).contiguous()
    return link_pos + quat_apply(link_quat, offsets_b)


def site_lin_vel_w(
    env: "ManagerBasedRlEnv",
    indices: list[int],
    offsets: torch.Tensor | None,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """World-frame linear velocity at each selected link's site.

    ``v_site = v_link + ω × (R_link · offset_local)``. With ``offsets=None``
    this reduces to ``link_lin_vel_w`` — matches the pre-parity behaviour. ``indices`` index
    ``asset_cfg``'s articulation (``None`` → primary).

    Returns ``(B, F, 3)`` aligned with ``indices``.
    """
    rs = asset_state(env, asset_cfg)
    link_lin_vel = rs.link_lin_vel_w[:, indices]  # (B, F, 3)
    if offsets is None:
        return link_lin_vel
    link_quat = rs.link_quat_w[:, indices]  # (B, F, 4)
    link_ang_vel = rs.link_ang_vel_w[:, indices]  # (B, F, 3)
    offsets_b = offsets.unsqueeze(0).expand_as(link_lin_vel).contiguous()
    r_world = quat_apply(link_quat, offsets_b)
    return link_lin_vel + torch.cross(link_ang_vel, r_world, dim=-1)
