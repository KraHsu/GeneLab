"""Curriculum-showcase runner: prints per-env terrain_levels and spawn distribution.

To demonstrate **promotion**, the runner periodically teleports a slice of envs 1.5 m
in front of their spawn position. Those envs then trip the ``walked >
distance_threshold`` branch on the next auto-reset, gain a level, and re-spawn on the
next-harder row. Envs that are not teleported demote toward level 0 — that's also a
valid curriculum behaviour to see.
"""

from typing import TYPE_CHECKING

import torch

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


_TELEPORT_EVERY: int = 30
_HALF_TELEPORT: int = 8  # half of num_envs=16
_TELEPORT_DISTANCE: float = 6.0  # > distance_threshold=5.0 so promote fires


class CurriculumShowcaseRunner(ShowcaseRunner):
    """Run 400 control steps; teleport half the envs every 30 steps."""

    task_slug = "curriculum"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=30)

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        if step > 0 and step % _TELEPORT_EVERY == 0:
            self._teleport_half(env)
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    @staticmethod
    def _teleport_half(env: "ManagerBasedRlEnv") -> None:
        terrain = env.scene.terrain
        if terrain is None:
            return
        env_ids = torch.arange(_HALF_TELEPORT, device=env.device)
        spawn = terrain.spawn_pos[env_ids]
        new_pos = spawn.clone()
        new_pos[:, 1] += _TELEPORT_DISTANCE
        quat = torch.zeros(env_ids.numel(), 4, device=env.device)
        quat[:, 0] = 1.0
        zero_vel = torch.zeros(env_ids.numel(), 3, device=env.device)
        env.articulation.write_root_state(new_pos, quat, zero_vel, zero_vel, env_ids)

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        root = self.log_root()
        terrain = env.scene.terrain
        if terrain is None:
            return
        levels = terrain.terrain_levels.detach().cpu().tolist()
        histogram = [levels.count(i) for i in range(terrain.cfg.num_rows)]
        with (root / "levels.log").open("a", encoding="utf-8") as fh:
            fh.write(
                f"step={step:04d}  levels={levels}  histogram={histogram} "
                f"(rows 0..{terrain.cfg.num_rows - 1})\n"
            )
