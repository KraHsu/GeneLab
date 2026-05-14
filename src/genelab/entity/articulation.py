"""Genesis articulated-robot wrapper used by ``ManagerBasedRlEnv`` and ``InteractiveScene``.

Two-phase construction mirrors Isaac Lab / MjLab:

* ``spawn(gs_scene)`` attaches an MJCF morph BEFORE ``scene.build()``.
* ``bind(num_envs, device)`` enumerates joints / links and pushes default PD gains AFTER
  ``scene.build()``.

The wrapper owns its own cached per-step state (``RobotState``); MDP and sensor code reads
that state through the env via ``env.robot_state``.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.entity._torch import quat_rotate_inverse, to_tensor


@dataclass
class ArticulationCfg:
    """Articulated-robot description.

    ``mjcf_path`` is the absolute path to a MuJoCo XML file. Joint dictionaries are
    ``name -> value`` where the keys may be regex patterns; multiple keys may match the same
    joint, in which case the last match wins.
    """

    mjcf_path: str = ""
    init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    joint_kp: dict[str, float] = field(default_factory=dict)
    joint_kv: dict[str, float] = field(default_factory=dict)
    action_scale: dict[str, float] | float = 0.5
    foot_link_names: tuple[str, ...] = ()


class RobotState:
    """Cached per-step articulation state. Refreshed by ``Articulation.refresh``."""

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


class Articulation:
    """Genesis articulated-robot entity with Isaac-Lab-style accessors."""

    def __init__(self, cfg: ArticulationCfg, *, name: str = "robot") -> None:
        if not cfg.mjcf_path:
            raise ValueError(f"ArticulationCfg(name={name!r}).mjcf_path must be set")
        self.cfg = cfg
        self.name = name
        self._gs_handle: Any = None
        self._num_envs: int = 0
        self._device: str = "cpu"
        self._joint_names: list[str] = []
        self._link_names: list[str] = []
        self._num_dofs: int = 0
        self._num_links: int = 0
        self._actuated_dof_idx: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._default_joint_pos: torch.Tensor = torch.empty(0)
        self._joint_kp: torch.Tensor = torch.empty(0)
        self._joint_kv: torch.Tensor = torch.empty(0)
        self._action_scale: torch.Tensor = torch.empty(0)
        self._data: RobotState | None = None

    # ------------------------------------------------------------------ spawn / bind

    def spawn(self, gs_scene: Any) -> None:
        """Pre-build: attach the MJCF morph to a Genesis scene."""
        import genesis as gs  # type: ignore[import-not-found]

        morph = gs.morphs.MJCF(
            file=str(self.cfg.mjcf_path),
            pos=tuple(self.cfg.init_pos),
            quat=tuple(self.cfg.init_quat),
        )
        self._gs_handle = gs_scene.add_entity(morph)

    def bind(self, num_envs: int, device: str) -> None:
        """Post-build: introspect joints / links, build per-joint tensors, push PD gains."""
        if self._gs_handle is None:
            raise RuntimeError(f"Articulation(name={self.name!r}).bind called before spawn")
        self._num_envs = num_envs
        self._device = device
        self._enumerate_joints_and_links()
        self._build_pose_and_gain_tensors()
        self._data = RobotState(num_envs, self._num_dofs, self._num_links, device)
        self._apply_default_gains()

    # ------------------------------------------------------------------ internals

    def _enumerate_joints_and_links(self) -> None:
        # Genesis exposes every DoF — including the 6 from a floating base — so we keep both
        # the per-joint actuated index list and the global DoF count to align tensors.
        robot = self._gs_handle
        joints = getattr(robot, "joints", None) or []
        joint_names: list[str] = []
        actuated_dof_indices: list[int] = []
        for j in joints:
            name = getattr(j, "name", None) or str(j)
            n_dofs = int(getattr(j, "n_dofs", 1))
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
            num = int(getattr(robot, "n_dofs", 0)) - 6
            joint_names = [f"joint_{i}" for i in range(max(num, 0))]
            actuated_dof_indices = list(range(6, 6 + len(joint_names)))
        self._joint_names = joint_names
        self._num_dofs = len(joint_names)
        self._actuated_dof_idx = torch.tensor(
            actuated_dof_indices, dtype=torch.long, device=self._device
        )

        links = getattr(robot, "links", []) or []
        link_names = [getattr(link, "name", f"link_{i}") for i, link in enumerate(links)]
        self._link_names = link_names
        self._num_links = max(len(link_names), 1)

    def _build_pose_and_gain_tensors(self) -> None:
        cfg = self.cfg
        self._default_joint_pos = self.build_per_joint_tensor(cfg.default_joint_pos, default=0.0)
        self._joint_kp = self.build_per_joint_tensor(cfg.joint_kp, default=0.0)
        self._joint_kv = self.build_per_joint_tensor(cfg.joint_kv, default=0.0)
        if isinstance(cfg.action_scale, dict):
            self._action_scale = self.build_per_joint_tensor(cfg.action_scale, default=0.25)
        else:
            self._action_scale = torch.full(
                (self._num_dofs,), float(cfg.action_scale), device=self._device
            )

    def _apply_default_gains(self) -> None:
        robot = self._gs_handle
        set_kp = getattr(robot, "set_dofs_kp", None)
        set_kv = getattr(robot, "set_dofs_kv", None)
        if set_kp is not None and self._joint_kp.numel() > 0:
            set_kp(self._joint_kp, self._actuated_dof_idx)
        if set_kv is not None and self._joint_kv.numel() > 0:
            set_kv(self._joint_kv, self._actuated_dof_idx)

    # ------------------------------------------------------------------ public methods

    def build_per_joint_tensor(self, mapping: dict[str, float], *, default: float) -> torch.Tensor:
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

    def refresh(self) -> None:
        if self._data is None:
            return
        rs = self._data
        robot = self._gs_handle
        try:
            rs.root_pos = to_tensor(robot.get_pos(), self._device)
            rs.root_quat = to_tensor(robot.get_quat(), self._device)
            rs.root_lin_vel_w = to_tensor(robot.get_vel(), self._device)
            rs.root_ang_vel_w = to_tensor(robot.get_ang(), self._device)
            joint_pos_full = to_tensor(robot.get_dofs_position(), self._device)
            joint_vel_full = to_tensor(robot.get_dofs_velocity(), self._device)
            rs.joint_pos = joint_pos_full.index_select(-1, self._actuated_dof_idx)
            rs.joint_vel = joint_vel_full.index_select(-1, self._actuated_dof_idx)
        except AttributeError:
            pass
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
            tensor = to_tensor(value, self._device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(self._num_envs, -1, -1)
            setattr(rs, target, tensor)
        rs.root_lin_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_lin_vel_w)
        rs.root_ang_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_ang_vel_w)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand_as(rs.root_lin_vel_w)
        rs.projected_gravity_b = quat_rotate_inverse(rs.root_quat, gravity_w)

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write actuated-joint positions / velocities for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        robot = self._gs_handle
        set_pos = getattr(robot, "set_dofs_position", None)
        set_vel = getattr(robot, "set_dofs_velocity", None)
        # ``envs_idx=`` is a newer Genesis kwarg; fall back to non-batched API on older builds.
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

    def write_root_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write floating-base pose + velocity for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        robot = self._gs_handle
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

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset actuated joints to default pose + zero velocity for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        default = self._default_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1).contiguous()
        zeros_v = torch.zeros_like(default)
        self.write_joint_state(default, zeros_v, env_ids)

    # ------------------------------------------------------------------ properties

    @property
    def gs_handle(self) -> Any:
        return self._gs_handle

    @property
    def data(self) -> RobotState:
        if self._data is None:
            raise RuntimeError(f"Articulation(name={self.name!r}).data accessed before bind")
        return self._data

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def num_links(self) -> int:
        return self._num_links

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
    def actuated_dof_idx(self) -> torch.Tensor:
        return self._actuated_dof_idx

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
