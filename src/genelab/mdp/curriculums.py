"""Reusable curriculum term functions: adapt difficulty / spawn distribution per episode."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def terrain_levels_vel(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice | None,
    distance_threshold: float,
    demote_ratio: float = 0.5,
) -> torch.Tensor:
    """Promote / demote each env's terrain level by how far it walked from spawn.

    Computes the horizontal (XY) distance between the env's current root position and
    its last recorded spawn position. Envs that walked more than ``distance_threshold``
    move up one level (capped at ``cfg.num_rows - 1``); envs that walked less than
    ``distance_threshold * demote_ratio`` move down (floored at 0). Levels then drive a
    fresh spawn origin lookup on the importer, and the curriculum writes the new root
    pose into the sim so the next episode starts on the new sub-terrain.

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

    current_pos = env.articulation.data.root_pos[env_ids]
    spawn_pos = terrain.spawn_pos[env_ids]
    walked = torch.linalg.vector_norm(current_pos[:, :2] - spawn_pos[:, :2], dim=-1)

    move_up = walked > distance_threshold
    move_down = walked < distance_threshold * demote_ratio
    delta = move_up.long() - move_down.long()
    levels = terrain.terrain_levels
    new_levels = (levels[env_ids] + delta).clamp(0, terrain.cfg.num_rows - 1)
    levels[env_ids] = new_levels

    new_origins = terrain.update_env_origins(env_ids)
    n = int(env_ids.numel())
    zero_quat = torch.zeros(n, 4, device=env.device)
    zero_quat[:, 0] = 1.0
    zero_vel = torch.zeros(n, 3, device=env.device)
    env.articulation.write_root_state(new_origins, zero_quat, zero_vel, zero_vel, env_ids)

    return levels.float().mean()
