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
    """Configuration for a uniform-random body-frame velocity command.

    Each env is independently assigned to up to three orthogonal categories at each
    resample:

    * **standing** (``rel_standing_envs``): command is held at zero — the policy must
      learn to keep the base still.
    * **heading** (``rel_heading_envs``): ``ang_vel_z`` is recomputed every step from
      a PD on heading error (toward a randomly sampled target heading) instead of the
      raw sampled ``ang_vel_z``. Requires ``heading_command=True``.
    * **forward** (``rel_forward_envs``): sample is rewritten at resample time to a
      strict forward motion — ``vx = abs(sample).clamp(min=0.3)``, ``vy = 0``,
      ``ωz = 0``. Decouples translation learning from heading control; cf. mjlab's G1
      flat config uses 0.2.
    * **world** (``rel_world_envs``): the sampled velocity is interpreted in the world
      frame and rotated into the body frame every step via current yaw — i.e. the
      policy chases an absolute heading rather than an instantaneous body-frame
      command.

    The defaults ``(rel_standing_envs=0.0, rel_heading_envs=1.0, ...)`` preserve the
    pre-P2 behaviour where ``heading_command=True`` made every non-standing env a
    heading env. Set ``rel_heading_envs<1.0`` to admit envs that track the raw
    sampled ``ang_vel_z`` instead.
    """

    Ranges = _Ranges  # nested for ergonomics: `UniformVelocityCommandCfg.Ranges(...)`
    ranges: _Ranges = field(default_factory=_Ranges)
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    rel_forward_envs: float = 0.0
    rel_world_envs: float = 0.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    class_type: type[CommandTerm] | None = None  # filled by __post_init__

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = UniformVelocityCommand


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw angle (radians) from a wxyz quaternion ``(B, 4)``."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class UniformVelocityCommand(CommandTerm):
    """Per-env velocity command in the body frame.

    Layout of ``command``: ``[lin_vel_x, lin_vel_y, ang_vel_z]``. The four env-group
    flags described on :class:`UniformVelocityCommandCfg` are sampled independently;
    when more than one fires for the same env the per-step :meth:`_update_command`
    applies them in order *heading PD → world-frame rotation → standing zero-out* so
    later writes take precedence over earlier ones — same composition as mjlab.
    """

    cfg: UniformVelocityCommandCfg  # type: ignore[assignment]

    def __init__(self, cfg: UniformVelocityCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        # Snapshot of the resample-time velocity in the world frame; only read by
        # ``rel_world_envs`` envs so we can rotate it back into the body frame each
        # step. Stored even when no world envs exist — the extra (B, 3) buffer is
        # cheap and the branch keeps resample_command branch-free.
        self._command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._heading_target = torch.zeros(self.num_envs, device=self.device)
        self._is_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_forward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_world = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # When True, ``_resample_command`` becomes a no-op so play-time bridges
        # (keyboard / DearPyGui teleop) own ``_command`` exclusively. Toggle via
        # :py:meth:`set_external_source`.
        self._external_source: bool = False

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def is_standing_env(self) -> torch.Tensor:
        return self._is_standing

    @property
    def is_heading_env(self) -> torch.Tensor:
        return self._is_heading

    @property
    def is_forward_env(self) -> torch.Tensor:
        return self._is_forward

    @property
    def is_world_env(self) -> torch.Tensor:
        return self._is_world

    def set_external_source(self, enabled: bool) -> None:
        """Hand the command buffer over to (or back from) an external driver.

        Called by play-time bridges (keyboard / DearPyGui teleop) on attach so a
        per-step write to ``_command`` survives :py:meth:`compute` *and* any
        subsequent env-level reset. While enabled:

        * :py:meth:`_resample_command` becomes a no-op (the external driver owns
          ``_command``);
        * every env-group flag is cleared so :py:meth:`_update_command` is also a
          no-op — no heading PD, no world-frame rotation, no standing zero-out;
        * ``_command`` and ``_command_w`` are zeroed so the very first obs the
          policy sees matches the bridge's startup state (slider 0,0,0) rather
          than a random sample left over from the prior reset.

        Pass ``enabled=False`` to release the term back to its sampled behavior.
        """
        self._external_source = bool(enabled)
        if not self._external_source:
            return
        self._command[:] = 0.0
        self._command_w[:] = 0.0
        self._is_heading[:] = False
        self._is_standing[:] = False
        self._is_forward[:] = False
        self._is_world[:] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # External driver owns the buffer; freeze the term's state for the
        # subset being reset, leaving ``_command`` / group flags untouched.
        if self._external_source:
            return
        n = env_ids.numel()
        r = self.cfg.ranges
        # Sample all three velocity axes unconditionally — heading-mode envs will get
        # their ang_vel_z rewritten each step by the PD, and the sampled value is the
        # one consumed by non-heading envs.
        self._command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*r.lin_vel_x)
        self._command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*r.lin_vel_y)
        self._command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*r.ang_vel_z)

        # Heading-envs need a fresh target; the mask is rolled only when heading PD is on
        # so the per-step ``_update_command`` can skip the branch entirely otherwise.
        if self.cfg.heading_command:
            self._heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(*r.heading)
            self._is_heading[env_ids] = (
                torch.rand(n, device=self.device) < self.cfg.rel_heading_envs
            )
        else:
            self._is_heading[env_ids] = False

        self._is_standing[env_ids] = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        self._is_world[env_ids] = torch.rand(n, device=self.device) < self.cfg.rel_world_envs
        # Snapshot the body-frame sample as the world-frame reference for world envs.
        # Non-world envs never read this buffer, so we can write unconditionally.
        self._command_w[env_ids] = self._command[env_ids]

        # Forward envs: positive vx (≥0.3), zero vy + ωz. Applied at sample time so the
        # value the policy sees on the very first frame already reflects the constraint
        # — important when the policy reads commands as obs.
        self._is_forward[env_ids] = torch.rand(n, device=self.device) < self.cfg.rel_forward_envs
        fwd_local = self._is_forward[env_ids]
        fwd_ids = env_ids[fwd_local]
        if fwd_ids.numel() > 0:
            self._command[fwd_ids, 0] = self._command[fwd_ids, 0].abs().clamp(min=0.3)
            self._command[fwd_ids, 1] = 0.0
            self._command[fwd_ids, 2] = 0.0

        # Standing envs override everything; assign last so an env that's both standing
        # and forward ends up at zero (matches mjlab precedence — standing wins).
        # Clear both ``_command`` and ``_command_w`` so the cached world-frame command
        # also reads zero for standing envs (mjlab parity — ``velocity_command.py:137-138``).
        standing_ids = env_ids[self._is_standing[env_ids]]
        if standing_ids.numel() > 0:
            self._command[standing_ids] = 0.0
            self._command_w[standing_ids] = 0.0

    def _update_command(self) -> None:
        # 1. Heading PD overrides ωz for heading envs only.
        if self.cfg.heading_command and self._is_heading.any():
            quat = self._env.robot_state.root_quat
            yaw = _yaw_from_quat(quat)
            heading_error = torch.atan2(
                torch.sin(self._heading_target - yaw), torch.cos(self._heading_target - yaw)
            )
            h_ids = self._is_heading.nonzero(as_tuple=False).flatten()
            self._command[h_ids, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error[h_ids],
                self.cfg.ranges.ang_vel_z[0],
                self.cfg.ranges.ang_vel_z[1],
            )

        # 2. World envs: rotate the cached world-frame velocity into the body frame.
        if self._is_world.any():
            w_ids = self._is_world.nonzero(as_tuple=False).flatten()
            yaw_w = _yaw_from_quat(self._env.robot_state.root_quat[w_ids])
            cos_h = torch.cos(yaw_w)
            sin_h = torch.sin(yaw_w)
            vx_w = self._command_w[w_ids, 0]
            vy_w = self._command_w[w_ids, 1]
            self._command[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
            self._command[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

        # 3. Standing envs zero out — final assignment so it overrides heading/world.
        # mjlab parity (velocity_command.py:137-138): clear both buffers, not just the
        # body-frame one, so a standing env in world mode doesn't read a stale
        # ``_command_w`` value at the next ``_update_command`` step before resample.
        self._command[self._is_standing] = 0.0
        self._command_w[self._is_standing] = 0.0
