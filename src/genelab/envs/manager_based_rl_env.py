"""Genesis-backed manager-based RL environment.

This is a slim port of ``mjlab.envs.manager_based_rl_env`` adapted to Genesis. The env owns
a single articulated robot, a ground plane, and the seven manager-style MDP hooks.
"""

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from genelab.configs import ManagerBasedEnvCfg
from genelab.managers import (
    ActionManager,
    ActionTermCfg,
    CommandManager,
    CommandTermCfg,
    CurriculumManager,
    CurriculumTermCfg,
    EventManager,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)

if TYPE_CHECKING:
    pass


@dataclass
class RobotEntityCfg:
    """Robot description consumed by ``ManagerBasedRlEnv``.

    ``mjcf_path`` is the absolute path to a MuJoCo XML file. Joint dictionaries are name -> value
    where the keys may be regex patterns; multiple keys may match the same joint, in which case
    the last match wins.
    """

    mjcf_path: str = ""
    init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    joint_kp: dict[str, float] = field(default_factory=dict)
    joint_kv: dict[str, float] = field(default_factory=dict)
    action_scale: dict[str, float] | float = 0.5
    foot_link_names: tuple[str, ...] = ()


@dataclass
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
    """RL-flavored env config: composes per-manager term dictionaries.

    Backwards-compatible with ``ManagerBasedEnvCfg`` so existing tooling (``apply_overrides``,
    CLI listing) keeps working.
    """

    decimation: int = 4
    episode_length_s: float = 20.0
    device: str = "cuda"
    seed: int | None = None
    scale_rewards_by_dt: bool = True

    robot: RobotEntityCfg = field(default_factory=RobotEntityCfg)

    actions_cfg: dict[str, ActionTermCfg] = field(default_factory=dict)
    observations_cfg: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    rewards_cfg: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations_cfg: dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands_cfg: dict[str, CommandTermCfg] = field(default_factory=dict)
    events_cfg: dict[str, EventTermCfg] = field(default_factory=dict)
    curriculum_cfg: dict[str, CurriculumTermCfg] = field(default_factory=dict)


class RobotState:
    """Cached per-step robot state. Refreshed by ``ManagerBasedRlEnv._refresh_robot_state``."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        num_links: int,
        device: str,
    ) -> None:
        def z(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device)

        self.root_pos = z(num_envs, 3)
        self.root_quat = z(num_envs, 4)
        self.root_quat[:, 0] = 1.0
        self.root_lin_vel_w = z(num_envs, 3)
        self.root_ang_vel_w = z(num_envs, 3)
        self.root_lin_vel_b = z(num_envs, 3)
        self.root_ang_vel_b = z(num_envs, 3)
        self.projected_gravity_b = z(num_envs, 3)
        self.projected_gravity_b[:, 2] = -1.0
        self.joint_pos = z(num_envs, num_dofs)
        self.joint_vel = z(num_envs, num_dofs)
        self.link_pos = z(num_envs, num_links, 3)
        self.link_quat_w = z(num_envs, num_links, 4)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = z(num_envs, num_links, 3)
        self.link_ang_vel_w = z(num_envs, num_links, 3)


class ManagerBasedRlEnv:
    """Genesis-backed manager-based RL environment."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg) -> None:
        self.cfg = cfg
        self._device = cfg.device
        self._num_envs = max(1, int(cfg.scene.num_envs))
        self._dt_sim = float(cfg.scene.dt)
        self._decimation = int(cfg.decimation)
        self._step_dt = self._dt_sim * self._decimation
        self._max_episode_length = max(1, int(math.ceil(cfg.episode_length_s / self._step_dt)))
        self._build_scene()
        self._build_robot_introspection()
        self._robot_state = RobotState(
            self._num_envs, self._num_dofs, self._num_links, self._device
        )
        self._episode_length_buf = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._extras: dict[str, Any] = {}

        # Managers (order matters: actions before observations/rewards; commands before obs)
        self.action_manager = ActionManager(cfg.actions_cfg, self)
        self.command_manager = CommandManager(cfg.commands_cfg, self)
        self.observation_manager = ObservationManager(cfg.observations_cfg, self)
        self.reward_manager = RewardManager(
            cfg.rewards_cfg, self, scale_by_dt=cfg.scale_rewards_by_dt
        )
        self.termination_manager = TerminationManager(cfg.terminations_cfg, self)
        self.event_manager = EventManager(cfg.events_cfg, self)
        self.curriculum_manager = CurriculumManager(cfg.curriculum_cfg, self)

        # Apply PD gains, default pose, then run startup events.
        self._apply_default_gains()
        self.event_manager.apply("startup")
        self._refresh_robot_state()
        # Initial reset of all envs so the policy sees a well-defined obs.
        self.reset()

    # ------------------------------------------------------------------ properties

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def dt(self) -> float:
        return self._step_dt

    @property
    def physics_dt(self) -> float:
        return self._dt_sim

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @property
    def max_episode_length_s(self) -> float:
        return float(self.cfg.episode_length_s)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @property
    def num_actions(self) -> int:
        return int(self.action_manager.total_action_dim)

    @property
    def robot(self) -> Any:
        return self._robot

    @property
    def robot_state(self) -> RobotState:
        return self._robot_state

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def link_names(self) -> list[str]:
        return list(self._link_names)

    @property
    def body_names(self) -> list[str]:
        """Alias for ``link_names`` to match mjlab's terminology."""
        return list(self._link_names)

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env world-frame offset ``[num_envs, 3]``; zeros when Genesis uses local frames."""
        return self._env_origins

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._default_joint_pos

    @property
    def joint_kp(self) -> torch.Tensor:
        return self._joint_kp

    @property
    def joint_kv(self) -> torch.Tensor:
        return self._joint_kv

    @property
    def action_scale(self) -> torch.Tensor:
        return self._action_scale

    # ------------------------------------------------------------------ scene

    def _build_scene(self) -> None:
        """Initialize Genesis and build the parallel scene."""
        from genelab.rl.distributed import pin_cuda_device

        pinned = pin_cuda_device()
        if pinned is not None:
            # Distributed: force GPU backend and align self._device with rsl_rl's
            # expected cuda:{LOCAL_RANK} string. The CLI bootstrap has already set
            # CUDA_VISIBLE_DEVICES per rank and rewritten LOCAL_RANK to 0, so this
            # ends up as "cuda:0" — the rank's only visible device. That single
            # visible-device setup is what makes Quadrants (Genesis's compute
            # backend) allocate its tensors on the right physical GPU.
            self.cfg.scene.gpu = True
            self._device = pinned

        import genesis as gs  # type: ignore[import-not-found]

        gs_init = getattr(gs, "init", None)
        if gs_init is not None and not getattr(gs, "_initialized", False):
            backend = gs.gpu if self.cfg.scene.gpu else gs.cpu  # type: ignore[attr-defined]
            gs.init(backend=backend, logging_level="warning")

        sim_options = gs.options.SimOptions(
            dt=self._dt_sim,
            substeps=int(self.cfg.scene.substeps),
        )
        self._scene: Any = gs.Scene(
            sim_options=sim_options,
            show_viewer=self.cfg.scene.vis,
        )
        self._scene.add_entity(gs.morphs.Plane())
        robot_cfg = self.cfg.robot
        if not robot_cfg.mjcf_path:
            raise ValueError("ManagerBasedRlEnvCfg.robot.mjcf_path must be set")
        morph = gs.morphs.MJCF(
            file=str(robot_cfg.mjcf_path),
            pos=tuple(robot_cfg.init_pos),
            quat=tuple(robot_cfg.init_quat),
        )
        self._robot: Any = self._scene.add_entity(morph)
        self._scene.build(
            n_envs=self._num_envs,
            env_spacing=tuple(self.cfg.scene.env_spacing),
        )
        # Derive device from Genesis if available.
        gs_device = getattr(gs, "device", None)
        if gs_device is not None:
            try:
                self._device = str(gs_device())  # type: ignore[misc]
            except TypeError:
                self._device = str(gs_device)
        self._env_origins = self._compute_env_origins()

    def _compute_env_origins(self) -> torch.Tensor:
        """Per-env world-frame offset. Defaults to zeros; Genesis runs each env in its own frame."""
        scene_origins = getattr(self._scene, "envs_offset", None)
        if scene_origins is None:
            scene_origins = getattr(self._scene, "env_origins", None)
        if scene_origins is not None:
            return _to_tensor(scene_origins, self._device).reshape(self._num_envs, 3)
        return torch.zeros(self._num_envs, 3, device=self._device)

    def _build_robot_introspection(self) -> None:
        """Resolve joint / link names and default pose / gain tensors.

        Genesis exposes every DoF — including the 6 from the floating base — so we record both
        the per-joint actuated index list and the global DoF count to keep tensors aligned.
        """
        joints = getattr(self._robot, "joints", None) or []
        joint_names: list[str] = []
        actuated_dof_indices: list[int] = []
        for j in joints:
            name = getattr(j, "name", None) or str(j)
            n_dofs = int(getattr(j, "n_dofs", 1))
            # Skip the floating base joint (typically 6 DoFs, first in the chain).
            if n_dofs >= 6 and not joint_names:
                continue
            joint_names.append(name)
            dof_idx_local = getattr(j, "dofs_idx_local", None)
            if dof_idx_local is None:
                dof_start = int(getattr(j, "dof_start", len(actuated_dof_indices)))
                actuated_dof_indices.extend(range(dof_start, dof_start + n_dofs))
            else:
                actuated_dof_indices.extend(int(i) for i in dof_idx_local)
        if not joint_names:
            num = int(getattr(self._robot, "n_dofs", 0)) - 6
            joint_names = [f"joint_{i}" for i in range(max(num, 0))]
            actuated_dof_indices = list(range(6, 6 + len(joint_names)))
        self._joint_names = joint_names
        self._num_dofs = len(joint_names)
        self._actuated_dof_idx = torch.tensor(
            actuated_dof_indices, dtype=torch.long, device=self._device
        )
        total_dofs = int(getattr(self._robot, "n_dofs", self._num_dofs))
        self._total_dofs = total_dofs

        links = getattr(self._robot, "links", []) or []
        link_names = [getattr(link, "name", f"link_{i}") for i, link in enumerate(links)]
        self._link_names = link_names
        self._num_links = max(len(link_names), 1)

        cfg = self.cfg.robot
        self._default_joint_pos = self._build_per_joint_tensor(cfg.default_joint_pos, default=0.0)
        self._joint_kp = self._build_per_joint_tensor(cfg.joint_kp, default=0.0)
        self._joint_kv = self._build_per_joint_tensor(cfg.joint_kv, default=0.0)
        if isinstance(cfg.action_scale, dict):
            self._action_scale = self._build_per_joint_tensor(cfg.action_scale, default=0.25)
        else:
            self._action_scale = torch.full(
                (self._num_dofs,), float(cfg.action_scale), device=self._device
            )

    def _build_per_joint_tensor(self, mapping: dict[str, float], *, default: float) -> torch.Tensor:
        out = torch.full((self._num_dofs,), default, device=self._device)
        for pattern, value in mapping.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))
            for i, name in enumerate(self._joint_names):
                if regex.fullmatch(name) or regex.search(name):
                    out[i] = float(value)
        return out

    def _apply_default_gains(self) -> None:
        set_kp = getattr(self._robot, "set_dofs_kp", None)
        set_kv = getattr(self._robot, "set_dofs_kv", None)
        if set_kp is not None and self._joint_kp.numel() > 0:
            set_kp(self._joint_kp, self._actuated_dof_idx)
        if set_kv is not None and self._joint_kv.numel() > 0:
            set_kv(self._joint_kv, self._actuated_dof_idx)

    # ------------------------------------------------------------------ state

    def _refresh_robot_state(self) -> None:
        rs = self._robot_state
        robot = self._robot
        try:
            rs.root_pos = _to_tensor(robot.get_pos(), self._device)
            rs.root_quat = _to_tensor(robot.get_quat(), self._device)
            rs.root_lin_vel_w = _to_tensor(robot.get_vel(), self._device)
            rs.root_ang_vel_w = _to_tensor(robot.get_ang(), self._device)
            joint_pos_full = _to_tensor(robot.get_dofs_position(), self._device)
            joint_vel_full = _to_tensor(robot.get_dofs_velocity(), self._device)
            # Slice to actuated DoFs only.
            rs.joint_pos = joint_pos_full.index_select(-1, self._actuated_dof_idx)
            rs.joint_vel = joint_vel_full.index_select(-1, self._actuated_dof_idx)
        except AttributeError:
            pass
        # Per-link state — populated lazily, only when the Genesis getter is available.
        for attr, target in (
            ("get_links_pos", "link_pos"),
            ("get_links_quat", "link_quat_w"),
            ("get_links_vel", "link_lin_vel_w"),
            ("get_links_ang", "link_ang_vel_w"),
        ):
            getter = getattr(robot, attr, None)
            if getter is None:
                continue
            try:
                value = getter()
            except Exception:
                continue
            tensor = _to_tensor(value, self._device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(self._num_envs, -1, -1)
            setattr(rs, target, tensor)
        # Convert root-frame quantities to body frame for the policy.
        rs.root_lin_vel_b = _quat_rotate_inverse(rs.root_quat, rs.root_lin_vel_w)
        rs.root_ang_vel_b = _quat_rotate_inverse(rs.root_quat, rs.root_ang_vel_w)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand_as(rs.root_lin_vel_w)
        rs.projected_gravity_b = _quat_rotate_inverse(rs.root_quat, gravity_w)

    # ------------------------------------------------------------------ reference state

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write actuated-joint positions / velocities for ``env_ids`` (size ``[N, num_dofs]``)."""
        if env_ids.numel() == 0:
            return
        set_pos = getattr(self._robot, "set_dofs_position", None)
        set_vel = getattr(self._robot, "set_dofs_velocity", None)
        if set_pos is not None:
            try:
                set_pos(joint_pos, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_pos(joint_pos, self._actuated_dof_idx)
        if set_vel is not None:
            try:
                set_vel(joint_vel, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_vel(joint_vel, self._actuated_dof_idx)

    def write_root_state_to_sim(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write floating-base pose + velocity for ``env_ids`` (each size ``[N, 3 or 4]``)."""
        if env_ids.numel() == 0:
            return
        robot = self._robot
        for fn_name, value in (
            ("set_pos", root_pos),
            ("set_quat", root_quat),
            ("set_vel", root_lin_vel_w),
            ("set_ang", root_ang_vel_w),
        ):
            fn = getattr(robot, fn_name, None)
            if fn is None:
                continue
            try:
                fn(value, envs_idx=env_ids)
            except TypeError:
                fn(value)

    # ------------------------------------------------------------------ rollout

    def reset(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._reset_idx(env_ids)
        self._refresh_robot_state()
        obs = self.observation_manager.compute()
        return obs, dict(self._extras)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        # Reset joint state to default before events run, so events can layer randomisation.
        default = self._default_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1).contiguous()
        zeros_v = torch.zeros_like(default)
        set_dofs_pos = getattr(self._robot, "set_dofs_position", None)
        set_dofs_vel = getattr(self._robot, "set_dofs_velocity", None)
        if set_dofs_pos is not None:
            try:
                set_dofs_pos(default, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_dofs_pos(default, self._actuated_dof_idx)
        if set_dofs_vel is not None:
            try:
                set_dofs_vel(zeros_v, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_dofs_vel(zeros_v, self._actuated_dof_idx)
        # Run reset-mode events (root state randomisation, pushes, etc.).
        self.event_manager.apply("reset", env_ids)
        self.command_manager.reset(env_ids)
        self.action_manager.reset(env_ids)
        reward_extras = self.reward_manager.reset(env_ids)
        term_extras = self.termination_manager.reset(env_ids)
        curr_extras = self.curriculum_manager.compute(env_ids)
        self._episode_length_buf[env_ids] = 0
        self._extras["log"] = {**reward_extras, **term_extras, **curr_extras}

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action = action.to(self._device).clone()
        self.action_manager.process_action(action)
        for _ in range(self._decimation):
            self.action_manager.apply_action()
            self._scene.step()
        self._refresh_robot_state()
        self._episode_length_buf += 1
        # Resample commands and trigger interval-mode events.
        self.command_manager.compute(self._step_dt)
        self.event_manager.apply("interval", dt=self._step_dt)

        reward = self.reward_manager.compute(self._step_dt)
        dones = self.termination_manager.compute()
        time_outs = self.termination_manager.time_outs
        terminated = self.termination_manager.terminated

        # Auto-reset envs that hit done.
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self._refresh_robot_state()
        obs = self.observation_manager.compute()
        return obs, reward, terminated, time_outs, dict(self._extras)

    def close(self) -> None:
        scene = getattr(self, "_scene", None)
        if scene is None:
            return
        for attr in ("close", "stop", "destroy"):
            fn = getattr(scene, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return


def _to_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device, dtype=torch.float)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (world frame) into the body frame defined by ``quat`` (wxyz)."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)
