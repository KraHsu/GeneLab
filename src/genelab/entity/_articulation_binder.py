"""Articulation binding — the binding seam of ``Articulation``.

``ArticulationBinder`` runs the post-build introspection: joint / link enumeration,
joint-pattern matching, joint-limit caching, and actuator assembly. :meth:`bind`
returns a :class:`BindResult` that ``Articulation`` stores and exposes through its
properties. :func:`build_per_joint_tensor` is a module-level helper shared between
the binder (for the default-pose vector) and ``Articulation.build_per_joint_tensor``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from genelab.actuator import ActuatorBase

if TYPE_CHECKING:
    from genelab.entity.articulation import ArticulationCfg


def build_per_joint_tensor(
    joint_names: list[str], mapping: dict[str, float], *, default: float, device: str
) -> torch.Tensor:
    """Build a ``(num_dofs,)`` tensor from regex-keyed ``mapping`` over ``joint_names``."""
    out = torch.full((len(joint_names),), default, device=device)
    for pattern, value in mapping.items():
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        for i, name in enumerate(joint_names):
            if regex.fullmatch(name) or regex.search(name):
                out[i] = float(value)
    return out


@dataclass(frozen=True)
class BindResult:
    """The introspection / assembly outputs of :meth:`ArticulationBinder.bind`."""

    joint_names: list[str]
    link_names: list[str]
    num_dofs: int
    num_links: int
    actuated_dof_idx: torch.Tensor
    default_joint_pos: torch.Tensor
    joint_pos_limits: torch.Tensor
    joint_vel_limits: torch.Tensor
    action_scale_tensor: torch.Tensor
    actuators: dict[str, ActuatorBase]


class ArticulationBinder:
    """Post-build introspection + actuator assembly for ``Articulation`` (binding seam)."""

    def __init__(
        self,
        cfg: ArticulationCfg,
        name: str,
        gs_handle: Any,
        num_envs: int,
        device: str,
    ) -> None:
        self._cfg = cfg
        self._name = name
        self._gs_handle = gs_handle
        self._num_envs = num_envs
        self._device = device
        # Populated during bind so _match_joint_ids / _assemble_actuators can read them.
        self._joint_names: list[str] = []
        self._link_names: list[str] = []
        self._num_dofs: int = 0
        self._num_links: int = 0
        self._actuated_dof_idx: torch.Tensor = torch.empty(0, dtype=torch.long)

    def bind(self) -> BindResult:
        """Run enumeration, limit caching, and actuator assembly; return the outputs."""
        self._enumerate_joints_and_links()
        default_joint_pos = build_per_joint_tensor(
            self._joint_names, self._cfg.default_joint_pos, default=0.0, device=self._device
        )
        joint_pos_limits = self._compute_joint_pos_limits()
        joint_vel_limits = self._compute_joint_vel_limits()
        actuators, action_scale_tensor = self._assemble_actuators()
        return BindResult(
            joint_names=self._joint_names,
            link_names=self._link_names,
            num_dofs=self._num_dofs,
            num_links=self._num_links,
            actuated_dof_idx=self._actuated_dof_idx,
            default_joint_pos=default_joint_pos,
            joint_pos_limits=joint_pos_limits,
            joint_vel_limits=joint_vel_limits,
            action_scale_tensor=action_scale_tensor,
            actuators=actuators,
        )

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

    def _compute_joint_pos_limits(self) -> torch.Tensor:
        # Cache per-actuated-joint position limits as ``(num_joints, 2)`` (lower, upper).
        # Genesis returns one (lo, hi) per entity DoF including the 6 floating-base DoFs;
        # index by ``_actuated_dof_idx`` to keep only the joints the policy controls.
        # Base DoFs typically have ±inf limits, so omitting them keeps the reward finite.
        device = self._device
        get_dofs_limit = getattr(self._gs_handle, "get_dofs_limit", None)
        if get_dofs_limit is not None and self._actuated_dof_idx.numel() > 0:
            try:
                lower, upper = get_dofs_limit()
                lower = lower.to(device)
                upper = upper.to(device)
                # Some Genesis versions return per-env (n_envs, n_dofs); collapse if so.
                if lower.dim() == 2:
                    lower = lower[0]
                if upper.dim() == 2:
                    upper = upper[0]
                idx = self._actuated_dof_idx.to(device)
                return torch.stack([lower.index_select(0, idx), upper.index_select(0, idx)], dim=-1)
            except Exception:
                # Fallback (e.g. fake env in tests): wide-open limits so the reward
                # function still works without raising.
                return torch.tensor(
                    [[-float("inf"), float("inf")]] * self._actuated_dof_idx.numel(),
                    device=device,
                )
        return torch.empty(self._actuated_dof_idx.numel(), 2, device=device)

    def _compute_joint_vel_limits(self) -> torch.Tensor:
        # Per-actuated-joint velocity-limit magnitude (rad/s), shape ``(num_joints,)``.
        # Genesis has no velocity-limit getter, so it comes from the cfg: a uniform
        # ``joint_vel_limit`` broadcast across DoFs, or ``+∞`` (never trips) when unset.
        vel_limit = float("inf") if self._cfg.joint_vel_limit is None else self._cfg.joint_vel_limit
        return torch.full((self._actuated_dof_idx.numel(),), vel_limit, device=self._device)

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

    def _assemble_actuators(self) -> tuple[dict[str, ActuatorBase], torch.Tensor]:
        """Build :class:`ActuatorBase` instances from ``cfg.actuators``.

        Validates that the joint groups partition the articulation's actuated joints exactly:
        unmatched joints raise ``ValueError`` (forces the user to declare every joint, even
        passive ones with zero gains), and joints matched by more than one group also raise.
        """
        if not self._cfg.actuators:
            raise ValueError(
                f"Articulation(name={self._name!r}).cfg.actuators is empty; declare at least "
                f"one ImplicitPDActuatorCfg covering the actuated joints "
                f"({self._joint_names!r})"
            )
        coverage: dict[int, str] = {}
        actuators: dict[str, ActuatorBase] = {}
        action_scale_tensor = torch.full((self._num_dofs,), 1.0, device=self._device)
        for group_name, actuator_cfg in self._cfg.actuators.items():
            if actuator_cfg.class_type is None:
                raise ValueError(
                    f"Articulation(name={self._name!r}).cfg.actuators[{group_name!r}] has no "
                    f"class_type (use a concrete ActuatorBaseCfg subclass)"
                )
            joint_ids = self._match_joint_ids(actuator_cfg.target_names_expr)
            if joint_ids.numel() == 0:
                raise ValueError(
                    f"Articulation(name={self._name!r}).cfg.actuators[{group_name!r}] "
                    f"matched zero joints from patterns {actuator_cfg.target_names_expr!r}"
                )
            conflicts: list[tuple[str, str]] = []
            for jid in joint_ids.tolist():
                if jid in coverage:
                    conflicts.append((self._joint_names[jid], coverage[jid]))
            if conflicts:
                lines = ", ".join(f"{j!r} (already in {g!r})" for j, g in conflicts)
                raise ValueError(
                    f"Articulation(name={self._name!r}).cfg.actuators[{group_name!r}] "
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
                f"Articulation(name={self._name!r}): the following actuated joints are not "
                f"covered by any actuator group: {uncovered!r}. Declare a zero-gain "
                f"ImplicitPDActuatorCfg for passive joints if you want to leave them free."
            )
        return actuators, action_scale_tensor
