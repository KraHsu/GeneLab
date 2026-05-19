"""End-effector delta action term backed by damped-least-squares differential IK.

Per control step the policy emits an EE delta — ``(dx, dy, dz)`` (3-DoF) or
``(dx, dy, dz, droll, dpitch, dyaw)`` (6-DoF, axis-angle) — and the term solves
``Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx`` against the entity's spatial Jacobian to convert
that into a per-arm-joint delta. The result is added to the latest sensed joint
position and pushed through :meth:`Articulation.write_joint_targets_partial`,
the same dispatch path used by :class:`JointPositionAction`.

Mirrors the panda-gym / IsaacLab Cartesian action term, restricted to the
minimum useful surface — DLS damping is the only singularity handling; null-space
optimization and operational-space control are explicitly out of scope.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class DifferentialIKActionCfg(ActionTermCfg):
    """Cartesian end-effector delta action with damped-least-squares IK.

    ``body_name`` selects the link whose pose the policy controls (e.g. ``"hand"``
    for the Franka). ``joint_names`` regex patterns pick the arm joints the IK
    solver is allowed to drive — typically the 7 arm joints; gripper fingers are
    deliberately excluded and handled by a sibling term such as
    :class:`BinaryGripperAction`.

    ``scale`` is the per-step physical-delta cap applied to the (clipped) policy
    action: ``float`` scales every component identically (meters for position
    rows, radians for orientation rows), while a sequence must have length
    matching ``action_dim`` (3 or 6).

    ``use_orientation=False`` ships the position-only 3-DoF variant — the
    minimum useful version called out in #73 and a direct match for the
    panda-gym ``PandaPickAndPlace`` action surface. Setting it ``True`` opts in
    to 6-DoF axis-angle control.

    .. note::
       The robot's :class:`ArticulationCfg` must set
       ``requires_jac_and_ik=True`` so Genesis allocates the kinematic tape
       needed by ``RigidEntity.get_jacobian``. ``__init__`` raises a clear
       error if that flag was omitted.
    """

    body_name: str = ""
    joint_names: tuple[str, ...] = (".*",)
    scale: float | tuple[float, ...] = 0.05
    clip: tuple[float, float] | None = (-1.0, 1.0)
    use_orientation: bool = False
    damping: float = 0.05
    max_delta_joint: float = 0.2
    class_type: type[ActionTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = DifferentialIKAction


class DifferentialIKAction(ActionTerm):
    cfg: DifferentialIKActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: DifferentialIKActionCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        if not cfg.body_name:
            raise ValueError(
                "DifferentialIKActionCfg.body_name is required (e.g. 'hand' for Franka)"
            )

        articulation = env.articulation
        link_names = articulation.link_names
        try:
            self._ee_link_idx = link_names.index(cfg.body_name)
        except ValueError:
            raise ValueError(
                f"DifferentialIKAction: body_name {cfg.body_name!r} not in articulation "
                f"link list {link_names!r}"
            ) from None

        joint_names = env.joint_names
        matched: list[int] = []
        for pat in cfg.joint_names:
            try:
                regex = re.compile(pat)
            except re.error:
                regex = re.compile(re.escape(pat))
            for i, name in enumerate(joint_names):
                if (regex.fullmatch(name) or regex.search(name)) and i not in matched:
                    matched.append(i)
        if not matched:
            raise ValueError(
                f"DifferentialIKAction matched zero joints from patterns {cfg.joint_names!r}"
            )
        self._joint_indices = torch.tensor(matched, dtype=torch.long, device=self.device)
        # ``actuated_dof_idx`` maps each entry in ``joint_names`` to its full-entity
        # DoF index — the column index space Genesis's Jacobian uses. Indexing it
        # by ``_joint_indices`` gives the Jacobian columns this term controls.
        self._dof_columns = articulation.actuated_dof_idx.to(self.device).index_select(
            0, self._joint_indices
        )

        self._dim = 6 if cfg.use_orientation else 3
        scale = cfg.scale
        if isinstance(scale, (tuple, list)):
            if len(scale) != self._dim:
                raise ValueError(
                    f"DifferentialIKActionCfg.scale length {len(scale)} does not match "
                    f"action_dim {self._dim}"
                )
            self._scale = torch.tensor(scale, dtype=torch.float, device=self.device)
        else:
            self._scale = torch.full((self._dim,), float(scale), device=self.device)

        if cfg.damping <= 0:
            raise ValueError(f"DifferentialIKActionCfg.damping must be > 0 (got {cfg.damping})")
        # DLS damping is applied as ``J Jᵀ + λ² I`` — store ``λ²`` so the per-step
        # cost stays a single ``mat + scalar * I`` add.
        self._damping_sq = float(cfg.damping) ** 2

        self._raw = torch.zeros(self.num_envs, self._dim, device=self.device)
        self._target_q = torch.zeros(self.num_envs, len(matched), device=self.device)
        # Pre-build identity for the DLS regularizer so it never re-allocates.
        self._eye_rows = torch.eye(self._dim, device=self.device)

        # Cache joint-pos limits for the controlled joints. ``±inf`` rows are
        # silently treated as no-op by ``torch.clamp``, so wide-open limits
        # (e.g. tests using the fake env) flow through unchanged.
        limits = articulation.joint_pos_limits
        if limits.numel() == 0:
            self._joint_pos_lower: torch.Tensor | None = None
            self._joint_pos_upper: torch.Tensor | None = None
        else:
            limits = limits.to(self.device).index_select(0, self._joint_indices)
            self._joint_pos_lower = limits[:, 0]
            self._joint_pos_upper = limits[:, 1]

        gs_handle = articulation.gs_handle
        if gs_handle is None:
            raise RuntimeError(
                "DifferentialIKAction requires the articulation to be bound before "
                "construction (env.articulation.gs_handle is None)."
            )
        if not getattr(articulation.cfg, "requires_jac_and_ik", False):
            raise RuntimeError(
                "DifferentialIKAction requires ArticulationCfg.requires_jac_and_ik=True "
                "so Genesis allocates the Jacobian tape on the morph; "
                f"current cfg has it disabled for asset {articulation.name!r}."
            )
        self._gs_handle = gs_handle
        # ``RigidLink`` itself is what ``get_jacobian`` expects, not its index.
        self._ee_link = articulation.gs_handle.links[self._ee_link_idx]

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.clip is not None:
            lo, hi = self.cfg.clip
            actions = actions.clamp(min=lo, max=hi)
        self._raw[:] = actions

        # Desired spatial delta ``Δx`` in world frame (pos rows in metres,
        # orientation rows in radians as axis-angle).
        delta_x = actions * self._scale.unsqueeze(0)  # (B, dim)

        # Full ``(B, 6, n_full_dofs)`` Jacobian from Genesis; slice rows for the
        # active position/orientation dims and columns for the controlled joints.
        jac_full = self._gs_handle.get_jacobian(self._ee_link)
        if jac_full.dim() == 2:
            # Single-env Genesis builds drop the leading batch axis; broadcast
            # back so the algebra below stays batched.
            jac_full = jac_full.unsqueeze(0).expand(self.num_envs, -1, -1)
        jac = jac_full[:, : self._dim, :].index_select(-1, self._dof_columns)

        # DLS: Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx. ``torch.linalg.solve`` operates on the
        # batched square system, avoiding an explicit inverse.
        jjt = jac @ jac.transpose(-1, -2)
        reg = jjt + self._damping_sq * self._eye_rows.unsqueeze(0)
        rhs = delta_x.unsqueeze(-1)  # (B, dim, 1)
        solved = torch.linalg.solve(reg, rhs)
        delta_q = (jac.transpose(-1, -2) @ solved).squeeze(-1)  # (B, n_joints)

        # Clip per-step joint delta to keep the linearization local.
        bound = float(self.cfg.max_delta_joint)
        delta_q = delta_q.clamp(min=-bound, max=bound)

        current_q = self._env.robot_state.joint_pos.index_select(1, self._joint_indices)
        target_q = current_q + delta_q
        if self._joint_pos_lower is not None and self._joint_pos_upper is not None:
            target_q = torch.maximum(target_q, self._joint_pos_lower.unsqueeze(0))
            target_q = torch.minimum(target_q, self._joint_pos_upper.unsqueeze(0))
        self._target_q[:] = target_q

    def apply_actions(self) -> None:
        self._env.articulation.write_joint_targets_partial(self._joint_indices, self._target_q)
