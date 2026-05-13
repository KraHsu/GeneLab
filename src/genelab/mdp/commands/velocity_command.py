"""Uniform random twist (lin_vel_x, lin_vel_y, ang_vel_z) command generator."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class _Ranges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    heading: tuple[float, float] = (-math.pi, math.pi)


@dataclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for a uniform-random body-frame velocity command."""

    Ranges = _Ranges  # nested for ergonomics: `UniformVelocityCommandCfg.Ranges(...)`
    ranges: _Ranges = field(default_factory=_Ranges)
    rel_standing_envs: float = 0.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    class_type: type[CommandTerm] | None = None  # filled by __post_init__

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = UniformVelocityCommand


class UniformVelocityCommand(CommandTerm):
    """Per-env velocity command in the body frame.

    Layout of ``command``: ``[lin_vel_x, lin_vel_y, ang_vel_z]``. When ``heading_command`` is set,
    ``ang_vel_z`` is overwritten each step to drive the body toward a randomized heading.
    """

    cfg: UniformVelocityCommandCfg  # type: ignore[assignment]

    def __init__(self, cfg: UniformVelocityCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._heading_target = torch.zeros(self.num_envs, device=self.device)
        self._is_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        r = self.cfg.ranges
        self._command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*r.lin_vel_x)
        self._command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*r.lin_vel_y)
        if self.cfg.heading_command:
            self._heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(*r.heading)
        else:
            self._command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*r.ang_vel_z)
        # Pick standing envs.
        standing = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        self._is_standing[env_ids] = standing
        self._command[env_ids[standing]] = 0.0

    def _update_command(self) -> None:
        if not self.cfg.heading_command:
            return
        quat = self._env.robot_state.root_quat
        # Yaw from wxyz quaternion.
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        heading_error = torch.atan2(
            torch.sin(self._heading_target - yaw), torch.cos(self._heading_target - yaw)
        )
        self._command[:, 2] = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
        )
        self._command[self._is_standing] = 0.0


