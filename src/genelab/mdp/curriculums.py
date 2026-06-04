"""Reusable curriculum term functions: adapt difficulty / spawn distribution per episode."""

from typing import TYPE_CHECKING, Any

import torch

from genelab.mdp._helpers import asset_articulation

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def terrain_levels_vel(
    env: "EnvContext",
    env_ids: torch.Tensor | slice | None,
    distance_threshold: float,
    command_name: str = "twist",
    demote_ratio: float = 0.5,
    spawn_clearance: float = 0.1,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """Promote / demote each env's terrain level by how far it walked from spawn.

    Computes the horizontal (XY) distance between the env's current root position and
    its last recorded spawn position. Envs that walked more than ``distance_threshold``
    move up one level (capped at ``cfg.num_rows - 1``); envs that covered less than
    ``demote_ratio`` of the distance they were *commanded* to walk (command speed ×
    episode length) move down (floored at 0), so the curriculum self-balances at each
    policy's capability. Levels then drive a fresh spawn origin lookup on the importer,
    and the curriculum writes the new root pose into the sim so the next episode starts
    on the new sub-terrain.

    The new origins carry only the planar cell center (``z`` = terrain root height), so
    the base is reseated at the sampled surface height plus the asset's standing height
    (``init_pos[2]``) plus ``spawn_clearance``. Writing the bare origin would drop the
    base to ``z = 0`` and bury the robot in the height field, whose interpenetration
    explodes the contact solver (NaN at large ``dt``, an unconverging grind at small).

    No-op when ``env.scene.terrain`` is ``None``.

    Returns the mean terrain level across all envs, so the curriculum manager logs it
    under ``"Curriculum/<term-name>"`` automatically.
    """
    terrain = env.scene.terrain
    if terrain is None:
        return torch.zeros((), device=env.device)
    # Curriculum manager passes ``None`` on the first frame and a 1D tensor on resets;
    # ``slice`` is accepted for parity with the manager's type signature.
    if env_ids is None or isinstance(env_ids, slice):
        return terrain.terrain_levels.float().mean()
    if env_ids.numel() == 0:
        return terrain.terrain_levels.float().mean()

    articulation = asset_articulation(env, asset_cfg)
    current_pos = articulation.data.root_pos[env_ids]
    spawn_pos = terrain.spawn_pos[env_ids]
    walked = torch.linalg.vector_norm(current_pos[:, :2] - spawn_pos[:, :2], dim=-1)

    move_up = walked > distance_threshold
    # Isaac-Lab-aligned capability-relative demote: an env that covered less than half the
    # distance it was *commanded* to walk (i.e. failed to track on harder terrain) drops a
    # level, so the curriculum self-balances at each policy's capability instead of pushing
    # every env to the top.
    commanded = env.command_manager.get_command(command_name)[env_ids, :2]
    commanded_dist = torch.linalg.vector_norm(commanded, dim=-1) * float(
        getattr(env, "max_episode_length_s", 1.0) or 1.0
    )
    move_down = (walked < commanded_dist * demote_ratio) & ~move_up
    delta = move_up.long() - move_down.long()
    levels = terrain.terrain_levels
    new_levels = (levels[env_ids] + delta).clamp(0, terrain.cfg.num_rows - 1)
    levels[env_ids] = new_levels

    new_origins = terrain.update_env_origins(env_ids)
    stand_height = float(articulation.cfg.init_pos[2])
    root_pos = new_origins.clone()
    root_pos[:, 2] = terrain.surface_height_at(new_origins) + stand_height + spawn_clearance
    n = int(env_ids.numel())
    zero_quat = torch.zeros(n, 4, device=env.device)
    zero_quat[:, 0] = 1.0
    zero_vel = torch.zeros(n, 3, device=env.device)
    articulation.write_root_state(root_pos, zero_quat, zero_vel, zero_vel, env_ids)

    return levels.float().mean()


def commands_vel(
    env: "EnvContext",
    env_ids: torch.Tensor | slice | None,
    command_name: str,
    velocity_stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Stage-wise expansion of a velocity command's per-axis ranges.

    Each stage is ``{"step": int, "lin_vel_x": (lo, hi)?, "lin_vel_y": (lo, hi)?,
    "ang_vel_z": (lo, hi)?}``. The latest stage whose ``step <= env.common_step_counter``
    wins; absent axes inherit the previously-applied range. Mutates
    ``cfg.commands[command_name].ranges`` in place so the next ``_resample_command``
    samples from the new range.

    Mirrors ``tasks.velocity.mdp.curriculums.commands_vel``.
    """
    del env_ids  # ranges are global, not per-env
    term = env.command_manager.get_term(command_name)
    ranges = term.cfg.ranges  # type: ignore[attr-defined]
    counter = env.common_step_counter
    for stage in velocity_stages:
        if counter < stage["step"]:
            continue
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            if axis in stage and stage[axis] is not None:
                setattr(ranges, axis, tuple(stage[axis]))
    return {
        "lin_vel_x_min": torch.tensor(ranges.lin_vel_x[0]),
        "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1]),
        "lin_vel_y_min": torch.tensor(ranges.lin_vel_y[0]),
        "lin_vel_y_max": torch.tensor(ranges.lin_vel_y[1]),
        "ang_vel_z_min": torch.tensor(ranges.ang_vel_z[0]),
        "ang_vel_z_max": torch.tensor(ranges.ang_vel_z[1]),
    }
