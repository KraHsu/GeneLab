"""Genesis articulated-robot wrapper used by ``ManagerBasedRlEnv`` and ``InteractiveScene``.

Two-phase construction mirrors Isaac Lab / MjLab:

* ``spawn(gs_scene)`` attaches an MJCF morph BEFORE ``scene.build()``.
* ``bind(num_envs, device)`` enumerates joints / links, assembles actuators, and pushes the
  static actuator parameters to Genesis AFTER ``scene.build()``.

The wrapper owns its own cached per-step state (``RobotState``); MDP and sensor code reads
that state through the env via ``env.robot_state``.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.actuator import ActuatorBase, ActuatorBaseCfg
from genelab.entity._torch import quat_rotate_inverse, to_tensor


@dataclass
class ArticulationCfg:
    """Articulated-robot description.

    ``mjcf_path`` is the absolute path to a MuJoCo XML file. ``default_joint_pos`` keys are
    regex patterns against the articulated joint names (last-match wins). ``actuators`` is a
    dict of named actuator-group configs that must collectively cover every actuated joint:
    unmatched or duplicated joints raise ``ValueError`` at ``bind`` time.
    """

    mjcf_path: str = ""
    init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    actuators: dict[str, ActuatorBaseCfg] = field(default_factory=dict)
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
        self._action_scale_tensor: torch.Tensor = torch.empty(0)
        self._actuators: dict[str, ActuatorBase] = {}
        self._joint_pos_target: torch.Tensor = torch.empty(0)
        self._gravity_w: torch.Tensor = torch.empty(0)
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
        """Post-build: introspect joints / links, assemble actuators, push static params."""
        if self._gs_handle is None:
            raise RuntimeError(f"Articulation(name={self.name!r}).bind called before spawn")
        self._num_envs = num_envs
        self._device = device
        self._enumerate_joints_and_links()
        self._default_joint_pos = self.build_per_joint_tensor(
            self.cfg.default_joint_pos, default=0.0
        )
        self._joint_pos_target = (
            self._default_joint_pos.unsqueeze(0).expand(num_envs, -1).contiguous()
        )
        self._data = RobotState(num_envs, self._num_dofs, self._num_links, device)
        # Cached constant gravity vector, broadcast across envs. Built once at bind to keep
        # ``refresh`` allocation-free on the hot path (called every step + every reset).
        self._gravity_w = torch.zeros(num_envs, 3, device=device)
        self._gravity_w[:, 2] = -1.0
        self._assemble_actuators()

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

    def _match_joint_ids(self, patterns: tuple[str, ...]) -> torch.Tensor:
        matched: list[int] = []
        for pattern in patterns:
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))
            for i, name in enumerate(self._joint_names):
                if (regex.fullmatch(name) or regex.search(name)) and i not in matched:
                    matched.append(i)
        return torch.tensor(matched, dtype=torch.long, device=self._device)

    def _assemble_actuators(self) -> None:
        """Build :class:`ActuatorBase` instances from ``cfg.actuators``.

        Validates that the joint groups partition the articulation's actuated joints exactly:
        unmatched joints raise ``ValueError`` (forces the user to declare every joint, even
        passive ones with zero gains), and joints matched by more than one group also raise.
        """
        if not self.cfg.actuators:
            raise ValueError(
                f"Articulation(name={self.name!r}).cfg.actuators is empty; declare at least "
                f"one ImplicitPDActuatorCfg covering the actuated joints "
                f"({self._joint_names!r})"
            )
        coverage: dict[int, str] = {}
        actuators: dict[str, ActuatorBase] = {}
        action_scale_tensor = torch.full((self._num_dofs,), 1.0, device=self._device)
        for group_name, actuator_cfg in self.cfg.actuators.items():
            if actuator_cfg.class_type is None:
                raise ValueError(
                    f"Articulation(name={self.name!r}).cfg.actuators[{group_name!r}] has no "
                    f"class_type (use a concrete ActuatorBaseCfg subclass)"
                )
            joint_ids = self._match_joint_ids(actuator_cfg.target_names_expr)
            if joint_ids.numel() == 0:
                raise ValueError(
                    f"Articulation(name={self.name!r}).cfg.actuators[{group_name!r}] "
                    f"matched zero joints from patterns {actuator_cfg.target_names_expr!r}"
                )
            conflicts: list[tuple[str, str]] = []
            for jid in joint_ids.tolist():
                if jid in coverage:
                    conflicts.append((self._joint_names[jid], coverage[jid]))
            if conflicts:
                lines = ", ".join(f"{j!r} (already in {g!r})" for j, g in conflicts)
                raise ValueError(
                    f"Articulation(name={self.name!r}).cfg.actuators[{group_name!r}] "
                    f"conflicts on joints: {lines}"
                )
            for jid in joint_ids.tolist():
                coverage[jid] = group_name
            dof_ids = self._actuated_dof_idx.index_select(0, joint_ids)
            joint_names = [self._joint_names[i] for i in joint_ids.tolist()]
            actuator = actuator_cfg.class_type(
                actuator_cfg,
                name=group_name,
                joint_ids=joint_ids,
                dof_ids=dof_ids,
                joint_names=joint_names,
                num_envs=self._num_envs,
                device=self._device,
            )
            actuator.initialize(self._gs_handle)
            actuators[group_name] = actuator
            for jid in joint_ids.tolist():
                action_scale_tensor[jid] = float(actuator_cfg.action_scale)
        uncovered = [name for i, name in enumerate(self._joint_names) if i not in coverage]
        if uncovered:
            raise ValueError(
                f"Articulation(name={self.name!r}): the following actuated joints are not "
                f"covered by any actuator group: {uncovered!r}. Declare a zero-gain "
                f"ImplicitPDActuatorCfg for passive joints if you want to leave them free."
            )
        self._actuators = actuators
        self._action_scale_tensor = action_scale_tensor

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
            rs.root_pos.copy_(to_tensor(robot.get_pos(), self._device))
            rs.root_quat.copy_(to_tensor(robot.get_quat(), self._device))
            rs.root_lin_vel_w.copy_(to_tensor(robot.get_vel(), self._device))
            rs.root_ang_vel_w.copy_(to_tensor(robot.get_ang(), self._device))
            joint_pos_full = to_tensor(robot.get_dofs_position(), self._device)
            joint_vel_full = to_tensor(robot.get_dofs_velocity(), self._device)
            rs.joint_pos.copy_(joint_pos_full.index_select(-1, self._actuated_dof_idx))
            rs.joint_vel.copy_(joint_vel_full.index_select(-1, self._actuated_dof_idx))
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
            # Genesis may return either (num_envs, num_links, ...) or the bare (num_links, ...)
            # shape on older single-env builds; ``expand`` broadcasts without allocating.
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(self._num_envs, -1, -1)
            getattr(rs, target).copy_(tensor)
        rs.root_lin_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_lin_vel_w)
        rs.root_ang_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_ang_vel_w)
        rs.projected_gravity_b = quat_rotate_inverse(rs.root_quat, self._gravity_w)

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
        # Also reset the cached joint target so the next implicit-PD step uses the home pose.
        # ``index_copy_`` sidesteps Genesis's strict aliasing check on indexed assignment.
        self._joint_pos_target.index_copy_(0, env_ids.long(), default.clone())

    # ------------------------------------------------------------------ actuator routing

    def write_joint_targets_partial(
        self, local_joint_ids: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Stash a slice of joint position targets, then dispatch each actuator.

        ``local_joint_ids`` indexes into the articulation's actuated joint list (same space
        as :attr:`joint_names`). ``target`` has shape ``(num_envs, len(local_joint_ids))``.
        For every actuator group: implicit-PD actuators receive the slice via
        ``control_dofs_position``; force-channel actuators get their effort from
        :meth:`ActuatorBase.compute` and the result is pushed via ``control_dofs_force``.
        """
        if local_joint_ids.numel() == 0:
            return
        self._joint_pos_target.index_copy_(1, local_joint_ids, target)
        robot = self._gs_handle
        if self._data is None or robot is None:
            return
        for actuator in self._actuators.values():
            jids = actuator.joint_ids
            dofs = actuator.dof_ids
            target_slice = self._joint_pos_target.index_select(1, jids)
            if actuator.channel == "implicit_pd":
                ctrl = getattr(robot, "control_dofs_position", None) or getattr(
                    robot, "set_dofs_position_target", None
                )
                if ctrl is None:
                    continue
                try:
                    ctrl(target_slice, dofs)
                except TypeError:
                    ctrl(target_slice)
            else:
                joint_pos = self._data.joint_pos.index_select(1, jids)
                joint_vel = self._data.joint_vel.index_select(1, jids)
                effort = actuator.compute(joint_pos, joint_vel, target_slice)
                if effort is None:
                    continue
                ctrl = getattr(robot, "control_dofs_force", None)
                if ctrl is None:
                    continue
                try:
                    ctrl(effort, dofs)
                except TypeError:
                    ctrl(effort)

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
    def actuators(self) -> dict[str, ActuatorBase]:
        return dict(self._actuators)

    @property
    def action_scale_tensor(self) -> torch.Tensor:
        """Per-joint action scale aggregated across actuator groups (shape ``(num_dofs,)``)."""
        return self._action_scale_tensor
