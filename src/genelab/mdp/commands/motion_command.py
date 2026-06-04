"""Motion-imitation command term (BeyondMimic-style).

Port of ``tasks.tracking.mdp.commands.MotionCommand`` adapted to GeneLab's slim env.
The motion file is a numpy NPZ with the same schema the reference uses, so clips converted by
``scripts.csv_to_npz`` plug in directly:

- ``joint_pos``      : (T, num_dofs)
- ``joint_vel``      : (T, num_dofs)
- ``body_pos_w``     : (T, num_bodies, 3)
- ``body_quat_w``    : (T, num_bodies, 4) wxyz
- ``body_lin_vel_w`` : (T, num_bodies, 3)
- ``body_ang_vel_w`` : (T, num_bodies, 3)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from genelab.managers.command_manager import CommandTerm, CommandTermCfg
from genelab.mdp._helpers import resolve_articulation, resolve_robot_state
from genelab.utils.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


class MotionLoader:
    """Reads a motion NPZ into device tensors, sliced + permuted into the runtime order.

    ``body_indexes`` selects the (tracked-body) NPZ positions to keep; pass an
    explicit index tensor whose length equals ``len(cfg.body_names)``. ``joint_perm``
    optionally permutes the NPZ's joint axis into the robot's actuated-DoF order —
    leave it ``None`` when the NPZ already matches the robot positionally.
    """

    def __init__(
        self,
        motion_file: str,
        body_indexes: torch.Tensor,
        device: str | torch.device = "cpu",
        joint_perm: torch.Tensor | None = None,
    ) -> None:
        if not motion_file:
            raise ValueError("MotionLoader requires a non-empty motion_file path")
        data = np.load(motion_file)
        t = torch.float32
        joint_pos = torch.as_tensor(data["joint_pos"], dtype=t, device=device)
        joint_vel = torch.as_tensor(data["joint_vel"], dtype=t, device=device)
        if joint_perm is not None:
            joint_pos = joint_pos[:, joint_perm]
            joint_vel = joint_vel[:, joint_perm]
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=t, device=device)
        body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=t, device=device)
        body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=t, device=device)
        body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=t, device=device)
        self.body_pos_w = body_pos_w[:, body_indexes]
        self.body_quat_w = body_quat_w[:, body_indexes]
        self.body_lin_vel_w = body_lin_vel_w[:, body_indexes]
        self.body_ang_vel_w = body_ang_vel_w[:, body_indexes]
        self.time_step_total = int(self.joint_pos.shape[0])


@dataclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion-imitation command term.

    ``motion_body_order`` names the bodies stored along axis 1 of the NPZ in the
    file's native order — typically the reference's MJCF DFS traversal. When provided, it
    is used to map ``body_names`` to NPZ positions; when empty, the legacy
    positional-arange slicing is kept for clips that already match the robot order.

    ``motion_joint_order`` does the same for the NPZ's actuated joint axis; when
    empty the joint axis is assumed to already match the robot's joint order.
    """

    motion_file: str = ""
    anchor_body_name: str = ""
    body_names: tuple[str, ...] = ()
    motion_body_order: tuple[str, ...] = ()
    motion_joint_order: tuple[str, ...] = ()
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["start", "uniform"] = "uniform"
    class_type: type[CommandTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = MotionCommand


class MotionCommand(CommandTerm):
    """Drives the env toward a recorded motion clip, frame by frame.

    Exposes both reference (target) and current (robot) anchor + multi-body state, plus the
    ``body_pos_relative_w`` / ``body_quat_relative_w`` quantities used by the reference's relative-pose
    rewards. Sampling modes: ``"start"`` (always frame 0) and ``"uniform"`` (random frame).
    """

    cfg: MotionCommandCfg  # type: ignore[assignment]

    def __init__(self, cfg: MotionCommandCfg, env: "EnvContext") -> None:
        super().__init__(cfg, env)
        self._robot_state = resolve_robot_state(env, cfg.asset_name)
        articulation = resolve_articulation(env, cfg.asset_name)
        if not cfg.body_names:
            raise ValueError("MotionCommandCfg.body_names must be a non-empty tuple")
        if not cfg.anchor_body_name:
            raise ValueError("MotionCommandCfg.anchor_body_name must be set")

        robot_body_names = articulation.body_names
        try:
            self.robot_anchor_body_index = robot_body_names.index(cfg.anchor_body_name)
        except ValueError as exc:
            raise ValueError(
                f"anchor body {cfg.anchor_body_name!r} not found in robot bodies {robot_body_names!r}"
            ) from exc
        try:
            self.motion_anchor_body_index = list(cfg.body_names).index(cfg.anchor_body_name)
        except ValueError as exc:
            raise ValueError(
                f"anchor body {cfg.anchor_body_name!r} not in motion body_names {cfg.body_names!r}"
            ) from exc

        missing = [n for n in cfg.body_names if n not in robot_body_names]
        if missing:
            raise ValueError(
                f"motion body_names not present on robot: {missing}; available: {robot_body_names}"
            )
        self.body_indexes = torch.tensor(
            [robot_body_names.index(n) for n in cfg.body_names],
            dtype=torch.long,
            device=self.device,
        )

        if cfg.motion_body_order:
            try:
                motion_body_indexes = torch.tensor(
                    [cfg.motion_body_order.index(n) for n in cfg.body_names],
                    dtype=torch.long,
                    device="cpu",
                )
            except ValueError as exc:
                raise ValueError(
                    f"body in cfg.body_names missing from cfg.motion_body_order: {exc}"
                ) from exc
        else:
            motion_body_indexes = torch.arange(len(cfg.body_names), device="cpu")

        joint_perm: torch.Tensor | None = None
        if cfg.motion_joint_order:
            robot_joint_names = articulation.joint_names
            missing_joints = [n for n in robot_joint_names if n not in cfg.motion_joint_order]
            if missing_joints:
                raise ValueError(
                    f"motion_joint_order is missing robot joints: {missing_joints}; "
                    f"motion_joint_order={cfg.motion_joint_order!r}"
                )
            joint_perm = torch.tensor(
                [cfg.motion_joint_order.index(n) for n in robot_joint_names],
                dtype=torch.long,
                device="cpu",
            )

        self.motion = MotionLoader(
            cfg.motion_file,
            motion_body_indexes,
            device=self.device,
            joint_perm=joint_perm,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

    # ------------------------------------------------------------------ reference accessors

    @property
    def command(self) -> torch.Tensor:
        """Concatenated reference joint pos + vel — fed to policy as the command obs."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            + self._env.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    # ------------------------------------------------------------------ robot accessors

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self._robot_state.link_pos[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self._robot_state.link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._robot_state.link_pos[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._robot_state.link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._robot_state.link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._robot_state.link_ang_vel_w[:, self.body_indexes]

    # ------------------------------------------------------------------ sampling / resets

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = torch.randint(
            0, self.motion.time_step_total, (env_ids.numel(),), device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        else:
            self._uniform_sampling(env_ids)

        # Reference state at the (newly sampled) frame.
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        n = int(env_ids.numel())
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        pose_lo = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0))[0] for k in pose_keys], device=self.device
        )
        pose_hi = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0))[1] for k in pose_keys], device=self.device
        )
        pose_jitter = sample_uniform(pose_lo, pose_hi, (n, 6), device=self.device)
        root_pos = root_pos + pose_jitter[:, 0:3]
        delta_quat = quat_from_euler_xyz(pose_jitter[:, 3], pose_jitter[:, 4], pose_jitter[:, 5])
        root_ori = quat_mul(delta_quat, root_ori)

        vel_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        vel_lo = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0))[0] for k in vel_keys], device=self.device
        )
        vel_hi = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0))[1] for k in vel_keys], device=self.device
        )
        vel_jitter = sample_uniform(vel_lo, vel_hi, (n, 6), device=self.device)
        root_lin_vel = root_lin_vel + vel_jitter[:, :3]
        root_ang_vel = root_ang_vel + vel_jitter[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        jp_lo, jp_hi = self.cfg.joint_position_range
        if jp_lo != 0.0 or jp_hi != 0.0:
            joint_pos = joint_pos + sample_uniform(
                jp_lo, jp_hi, joint_pos.shape, device=self.device
            )

        self._env.write_root_state_to_sim(root_pos, root_ori, root_lin_vel, root_ang_vel, env_ids)
        self._env.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

    def update_relative_body_poses(self) -> None:
        """Recompute relative-body reference frames anchored at the robot's current root pose.

        Mirrors the reference so that relative-pose rewards / terminations compare bodies against the
        reference clip transformed into the robot's current anchor frame.
        """
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)

        # Yaw-only delta between robot anchor and motion anchor; share z from the motion clip.
        delta_pos_w = robot_anchor_pos_w.clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w
        )

    def _update_command(self) -> None:
        self.time_steps += 1
        wrapped = (self.time_steps >= self.motion.time_step_total).nonzero(as_tuple=False).flatten()
        if wrapped.numel() > 0:
            self._resample_command(wrapped)
        self.update_relative_body_poses()
