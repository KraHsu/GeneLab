"""Soft-terrain reward terms (paper §8.3, ADR-0001).

These read the privileged analytic deformable-terrain state
(:class:`genelab.terrains.DeformableTerrain`) and return a per-env positive cost; the env
cfg weights them negatively. They require ``cfg.deformable_terrain`` to be configured.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def _terrain_state(env: "EnvContext") -> object:
    driver = env.deformable_terrain
    if driver is None:
        raise RuntimeError("soft-terrain rewards require cfg.deformable_terrain to be set")
    return driver.terrain.state


def terrain_sinkage_l2(env: "EnvContext") -> torch.Tensor:
    """``Σ_foot depth²`` — penalise foot sinkage into the deformable terrain (paper §8.3).

    Encourages gaits that keep the feet near the surface (less sinkage) on soft ground.
    """
    depth = _terrain_state(env).depth.clamp_min(0.0)  # type: ignore[attr-defined]
    return torch.sum(depth * depth, dim=-1)


def footprint_revisit(env: "EnvContext") -> torch.Tensor:
    """``Σ_foot residual`` — penalise feet on high plastic-residual ground (paper §8.3).

    Discourages forming and lingering on deep footprints; with a spatial terrain map this
    becomes "don't step back into the holes you dug" (the paper's ``r_memory``).
    """
    residual = _terrain_state(env).residual.clamp_min(0.0)  # type: ignore[attr-defined]
    return torch.sum(residual, dim=-1)


def terrain_feet_air_time(
    env: "EnvContext",
    *,
    command_name: str,
    threshold_min: float = 0.1,
    threshold_max: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time gait shaper driven by the *terrain-derived* contact (depth > 0).

    Rewards each foot at touchdown for an air time in ``[threshold_min, threshold_max]`` —
    the standard ``feet_air_time`` reward, but using the analytic terrain's contact instead
    of a rigid contact sensor (which is blind under analytic support). Gated by command
    magnitude so it only fires when the robot is asked to move. This is what forces a real
    stepping gait instead of standing still on the frictionless-support floor.
    """
    state = _terrain_state(env)
    air_time = state.last_air_time.clamp(max=threshold_max)  # type: ignore[attr-defined]
    reward = torch.sum((air_time - threshold_min) * state.first_contact, dim=-1)  # type: ignore[attr-defined]
    command = env.command_manager.get_command(command_name)
    active = (command[:, :2].norm(dim=-1) > command_threshold).float()
    return reward * active


def terrain_feet_slip(
    env: "EnvContext",
    *,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """``Σ_foot ||v_xy||² · in_contact`` — penalise sliding a *planted* foot (terrain contact).

    Without this the robot "skates": it translates by dragging its feet via the analytic
    traction instead of stepping. Penalising horizontal velocity of feet that are in
    terrain contact (``depth > 0``) makes sliding costly, so the only cheap way to move is
    to lift and re-plant — i.e. take real steps. Gated by command magnitude.
    """
    driver = env.deformable_terrain
    if driver is None:
        raise RuntimeError("terrain_feet_slip requires cfg.deformable_terrain to be set")
    contact = driver.terrain.state.contact
    robot = env.articulations["robot"]
    idx = [robot.link_names.index(name) for name in driver.cfg.foot_link_names]
    foot_vel_xy = robot.data.link_lin_vel_w[:, idx, :2]
    slip = torch.sum(foot_vel_xy * foot_vel_xy, dim=-1) * contact
    command = env.command_manager.get_command(command_name)
    active = (command[:, :2].norm(dim=-1) > command_threshold).float()
    return torch.sum(slip, dim=-1) * active
