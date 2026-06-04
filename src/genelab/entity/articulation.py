"""Genesis articulated-robot wrapper used by ``ManagerBasedRlEnv`` and ``InteractiveScene``.

Two-phase construction mirrors Isaac Lab:

* ``spawn(gs_scene)`` attaches an MJCF morph BEFORE ``scene.build()``.
* ``bind(num_envs, device)`` enumerates joints / links, assembles actuators, and pushes the
  static actuator parameters to Genesis AFTER ``scene.build()``.

The wrapper owns its own cached per-step state (``RobotState``); MDP and sensor code reads
that state through the env via ``env.robot_state``.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.actuator import ActuatorBase, ActuatorBaseCfg
from genelab.entity._articulation_binder import (
    ArticulationBinder,
    build_per_joint_tensor as _build_per_joint_tensor,
)
from genelab.entity._articulation_state import (
    RobotState as RobotState,  # re-export: canonical home is _articulation_state
    ArticulationState,
)
from genelab.entity._articulation_writer import ArticulationWriter


@dataclass
class ArticulationCfg:
    """Articulated-robot description.

    Exactly one of ``mjcf_path`` / ``urdf_path`` must be set: ``mjcf_path`` points at a
    MuJoCo XML file, ``urdf_path`` at a URDF (Genesis 1.0 also accepts ``.xacro`` /
    ``.urdf.xacro`` files — the URDF morph preprocesses them via the ``xacro`` package
    at load time). ``xacro_args`` overrides ``xacro:arg`` declarations when loading from
    a xacro file; ignored for plain URDF and rejected for MJCF.

    ``default_joint_pos`` keys are regex patterns against the articulated joint names
    (last-match wins). ``actuators`` is a dict of named actuator-group configs that must
    collectively cover every actuated joint: unmatched or duplicated joints raise
    ``ValueError`` at ``bind`` time.
    """

    mjcf_path: str = ""
    urdf_path: str = ""
    xacro_args: dict[str, str] = field(default_factory=dict)
    init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    actuators: dict[str, ActuatorBaseCfg] = field(default_factory=dict)
    foot_link_names: tuple[str, ...] = ()
    # Optional uniform soft joint-velocity limit (rad/s, magnitude) applied across
    # all actuated DoFs. Genesis does not expose per-joint velocity limits, so this
    # is user-declared. ``None`` means "no limit" (+∞) — the velocity-limit
    # reward / termination (``mdp.joint_vel_limits`` / ``mdp.joint_vel_out_of_limit``)
    # then never trip. Opt-in; no behaviour change for existing tasks.
    joint_vel_limit: float | None = None
    # Opt-in flag for the Genesis morph's Jacobian / IK kinematic tape. Default
    # ``False`` because allocating those buffers has a cost and most tasks
    # control the arm via joint targets. Set ``True`` whenever an action term
    # such as ``DifferentialIKAction`` needs ``RigidEntity.get_jacobian``.
    requires_jac_and_ik: bool = False


class Articulation:
    """Genesis articulated-robot entity with Isaac-Lab-style accessors."""

    def __init__(self, cfg: ArticulationCfg, *, name: str = "robot") -> None:
        if bool(cfg.mjcf_path) == bool(cfg.urdf_path):
            raise ValueError(
                f"ArticulationCfg(name={name!r}): exactly one of mjcf_path / urdf_path "
                f"must be set (got mjcf_path={cfg.mjcf_path!r}, urdf_path={cfg.urdf_path!r})"
            )
        if cfg.xacro_args and not cfg.urdf_path:
            raise ValueError(
                f"ArticulationCfg(name={name!r}): xacro_args is only valid with urdf_path"
            )
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
        # ``(num_actuated_joints, 2)`` lower/upper joint-position limits sliced from
        # Genesis's per-DoF limits. Used by ``mdp.joint_pos_limits``. Populated in bind.
        self._joint_pos_limits: torch.Tensor = torch.empty(0, 2)
        self._action_scale_tensor: torch.Tensor = torch.empty(0)
        self._actuators: dict[str, ActuatorBase] = {}
        self._joint_pos_target: torch.Tensor = torch.empty(0)
        self._state: ArticulationState | None = None
        self._writer: ArticulationWriter | None = None

    # ------------------------------------------------------------------ spawn / bind

    def spawn(self, gs_scene: Any) -> None:
        """Pre-build: attach the robot morph (MJCF or URDF/xacro) to a Genesis scene.

        Dispatches on which of ``mjcf_path`` / ``urdf_path`` the cfg has set. Genesis 1.0's
        ``gs.morphs.URDF`` auto-preprocesses ``.xacro`` and ``.urdf.xacro`` files via the
        ``xacro`` package at load time, so an extension check at the GeneLab layer is
        unnecessary — the URDF morph handles both shapes.
        """
        import genesis as gs  # type: ignore[import-not-found]

        morph_kwargs: dict[str, Any] = dict(
            pos=tuple(self.cfg.init_pos),
            quat=tuple(self.cfg.init_quat),
        )
        if self.cfg.requires_jac_and_ik:
            morph_kwargs["requires_jac_and_IK"] = True
        if self.cfg.urdf_path:
            morph_kwargs["file"] = str(self.cfg.urdf_path)
            if self.cfg.xacro_args:
                morph_kwargs["xacro_args"] = dict(self.cfg.xacro_args)
            morph = gs.morphs.URDF(**morph_kwargs)
        else:
            morph_kwargs["file"] = str(self.cfg.mjcf_path)
            morph = gs.morphs.MJCF(**morph_kwargs)
        self._gs_handle = gs_scene.add_entity(morph)

    def bind(self, num_envs: int, device: str) -> None:
        """Post-build: introspect joints / links, assemble actuators, push static params."""
        if self._gs_handle is None:
            raise RuntimeError(f"Articulation(name={self.name!r}).bind called before spawn")
        self._num_envs = num_envs
        self._device = device
        # Binding seam: introspection + actuator assembly produce the metadata
        # the properties / state / writer read; store the results on the facade.
        result = ArticulationBinder(self.cfg, self.name, self._gs_handle, num_envs, device).bind()
        self._joint_names = result.joint_names
        self._link_names = result.link_names
        self._num_dofs = result.num_dofs
        self._num_links = result.num_links
        self._actuated_dof_idx = result.actuated_dof_idx
        self._default_joint_pos = result.default_joint_pos
        self._joint_pos_limits = result.joint_pos_limits
        self._joint_vel_limits = result.joint_vel_limits
        self._action_scale_tensor = result.action_scale_tensor
        self._actuators = result.actuators
        # ``.clone()`` guarantees independent storage so a later ``index_copy_`` on the target
        # never aliases ``_default_joint_pos`` (the num_envs=1 contiguous-alias bug — without
        # it ``joint_pos_rel`` obs read a corrupted reference at single-env batch sizes).
        self._joint_pos_target = self._default_joint_pos.unsqueeze(0).expand(num_envs, -1).clone()
        self._state = ArticulationState(
            self._gs_handle,
            num_envs=num_envs,
            num_dofs=self._num_dofs,
            num_links=self._num_links,
            device=device,
            actuated_dof_idx=self._actuated_dof_idx,
        )
        # Shares the joint-target buffer + actuator dict by reference (the write seam).
        self._writer = ArticulationWriter(
            self._gs_handle,
            actuated_dof_idx=self._actuated_dof_idx,
            default_joint_pos=self._default_joint_pos,
            joint_pos_target=self._joint_pos_target,
            actuators=self._actuators,
            device=self._device,
        )

    # ------------------------------------------------------------------ public methods

    def build_per_joint_tensor(self, mapping: dict[str, float], *, default: float) -> torch.Tensor:
        """Build a ``(num_dofs,)`` tensor from a regex-keyed ``mapping`` over the joint names."""
        return _build_per_joint_tensor(
            self._joint_names, mapping, default=default, device=self._device
        )

    def refresh(self) -> None:
        """Update the cached :class:`RobotState` in place from the Genesis handle.

        Delegates to :class:`ArticulationState` (the read seam); a no-op before
        :meth:`bind`.
        """
        if self._state is not None:
            self._state.refresh()

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write actuated-joint positions / velocities for ``env_ids``."""
        if self._writer is not None:
            self._writer.write_joint_state(joint_pos, joint_vel, env_ids)

    def write_root_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write floating-base pose + velocity for ``env_ids``."""
        if self._writer is not None:
            self._writer.write_root_state(
                root_pos, root_quat, root_lin_vel_w, root_ang_vel_w, env_ids
            )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset actuated joints to default pose + zero velocity for ``env_ids``."""
        if self._writer is not None:
            self._writer.reset(env_ids)

    def write_joint_targets_partial(
        self, local_joint_ids: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Stash a slice of joint position targets, then dispatch each actuator.

        Delegates to :class:`ArticulationWriter` (the write seam); a no-op before
        :meth:`bind`. ``local_joint_ids`` indexes the actuated joint list (same space as
        :attr:`joint_names`); ``target`` has shape ``(num_envs, len(local_joint_ids))``.
        """
        if self._writer is not None:
            self._writer.write_joint_targets_partial(local_joint_ids, target)

    # ------------------------------------------------------------------ properties

    @property
    def gs_handle(self) -> Any:
        return self._gs_handle

    @property
    def data(self) -> RobotState:
        if self._state is None:
            raise RuntimeError(f"Articulation(name={self.name!r}).data accessed before bind")
        return self._state.data

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
    def actuated_dof_ids(self) -> torch.Tensor:
        """Global Genesis DoF index for each actuated joint, in :attr:`joint_names` order.

        ``get_dofs_*`` returns every entity DoF (including the floating base); index by this
        to keep only the policy-controlled joints. Used by
        :class:`~genelab.sensor.ForceTorqueSensor` to slice ``get_dofs_force`` to its joints.
        """
        return self._actuated_dof_idx

    @property
    def link_names(self) -> list[str]:
        return list(self._link_names)

    @property
    def body_names(self) -> list[str]:
        """Alias for ``link_names`` to match Isaac Lab's terminology."""
        return list(self._link_names)

    @property
    def actuated_dof_idx(self) -> torch.Tensor:
        return self._actuated_dof_idx

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._default_joint_pos

    @property
    def joint_pos_limits(self) -> torch.Tensor:
        """Per-actuated-joint ``(lower, upper)`` position limits, shape ``(num_joints, 2)``.

        Read from Genesis at bind time via ``get_dofs_limit()`` and sliced to the
        actuated DoFs. Used by :func:`genelab.mdp.joint_pos_limits` to compute the
        per-joint excursion penalty against the real model limits rather than a
        flat ±π fallback.
        """
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> torch.Tensor:
        """Per-actuated-joint velocity-limit magnitude (rad/s), shape ``(num_joints,)``.

        Sourced from ``ArticulationCfg.joint_vel_limit`` (Genesis exposes no
        velocity limit); ``+∞`` when unset. Used by
        :func:`genelab.mdp.joint_vel_limits` and
        :func:`genelab.mdp.joint_vel_out_of_limit`.
        """
        return self._joint_vel_limits

    @property
    def actuators(self) -> dict[str, ActuatorBase]:
        return dict(self._actuators)

    @property
    def action_scale_tensor(self) -> torch.Tensor:
        """Per-joint action scale aggregated across actuator groups (shape ``(num_dofs,)``)."""
        return self._action_scale_tensor
