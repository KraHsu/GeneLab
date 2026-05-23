"""Joint force-torque sensor (ROADMAP M3.5).

Reports each selected joint's internal reaction force/torque — Genesis's
``get_dofs_force`` (the total internal DoF force: constraint + actuation), as opposed
to the *commanded* torque (``get_dofs_control_force``) cached in
``RobotState.applied_torque``. For revolute / prismatic joints this is the scalar
reaction about / along the joint axis (the "joint-FT" of ROADMAP M3.5; a full 6-axis
wrench / fingertip pressure array is deferred).
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_handle
from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class ForceTorqueData:
    """Per-step joint reaction force/torque. ``force`` is ``(num_envs, num_joints)``."""

    force: torch.Tensor


@dataclass
class ForceTorqueSensorCfg(SensorCfg):
    """Monitor per-joint internal reaction force/torque.

    Select joints with an explicit ``joint_names`` tuple or a ``joint_names_expr`` regex
    (matched against ``env.joint_names`` with ``re.search``). With neither, every actuated
    joint is monitored.
    """

    joint_names: tuple[str, ...] = ()
    joint_names_expr: str | None = None

    def build(self) -> "ForceTorqueSensor":
        return ForceTorqueSensor(self)


def _resolve_joint_indices(
    cfg: ForceTorqueSensorCfg, joint_names: list[str]
) -> tuple[list[int], list[str]]:
    if cfg.joint_names_expr is not None:
        pattern = re.compile(cfg.joint_names_expr)
        matched = [(i, n) for i, n in enumerate(joint_names) if pattern.search(n)]
    elif cfg.joint_names:
        missing = [n for n in cfg.joint_names if n not in joint_names]
        if missing:
            raise ValueError(
                f"ForceTorqueSensorCfg(name={cfg.name!r}): joint(s) {missing!r} not in env.joint_names"
            )
        order = {n: i for i, n in enumerate(joint_names)}
        matched = [(order[n], n) for n in cfg.joint_names]
    else:
        matched = list(enumerate(joint_names))  # default: every actuated joint
    if not matched:
        raise ValueError(
            f"ForceTorqueSensorCfg(name={cfg.name!r}): "
            f"joint_names_expr={cfg.joint_names_expr!r} matched nothing"
        )
    return [i for i, _ in matched], [n for _, n in matched]


class ForceTorqueSensor(Sensor[ForceTorqueData]):
    """Per-joint reaction force/torque from Genesis's ``get_dofs_force``."""

    def __init__(self, cfg: ForceTorqueSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._dof_ids: torch.Tensor | None = None  # global DoF id per selected joint
        self._resolved_joint_names: list[str] = []

    @property
    def joint_names(self) -> list[str]:
        return list(self._resolved_joint_names)

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        local, names = _resolve_joint_indices(
            self._cfg_typed, entity_articulation(env, self._cfg.entity_name).joint_names
        )
        self._resolved_joint_names = names
        local_t = torch.tensor(local, dtype=torch.long, device=env.device)
        # Map joint-list indices → global Genesis DoF ids (``get_dofs_force`` returns
        # every entity DoF, including the floating base).
        actuated = entity_articulation(env, self._cfg.entity_name).actuated_dof_ids.to(env.device)
        self._dof_ids = actuated.index_select(0, local_t)

    def _compute_data(self) -> ForceTorqueData:
        env = self._env
        assert env is not None and self._dof_ids is not None
        n = self._dof_ids.numel()
        zeros = torch.zeros(env.num_envs, n, device=env.device)
        getter = getattr(entity_handle(env, self._cfg.entity_name), "get_dofs_force", None)
        if getter is None:
            return ForceTorqueData(force=zeros)  # Genesis fake / not built — keep usable in tests.
        try:
            force = getter()
        except Exception:
            return ForceTorqueData(force=zeros)
        force = force.to(env.device)
        if force.dim() == 1:
            force = force.unsqueeze(0).expand(env.num_envs, -1)
        return ForceTorqueData(force=force.index_select(-1, self._dof_ids))
