"""Single-dim binary gripper action term — maps a scalar policy output to an
``{open, closed}`` finger-joint target broadcast across the matched joints.

Companions :class:`DifferentialIKAction` to reproduce panda-gym's 4-DoF
``(dx, dy, dz, gripper)`` Cartesian action space without entangling the IK
solver with the gripper degree of freedom.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm, ActionTermCfg
from genelab.mdp._helpers import resolve_articulation
from genelab.mdp.actions._joint_match import match_joints

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class BinaryGripperActionCfg(ActionTermCfg):
    """One-dim binary gripper action.

    The policy emits a single scalar per env; values strictly greater than
    ``threshold`` snap the matched finger joints to ``open_pos``, the rest to
    ``closed_pos``. Defaults match the Franka Panda's ``finger_joint*`` range
    (``0.0`` closed → ``0.04`` open). Use one regex per finger group via
    ``joint_names``.
    """

    joint_names: tuple[str, ...] = (".*",)
    open_pos: float = 0.04
    closed_pos: float = 0.0
    threshold: float = 0.0
    class_type: type[ActionTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = BinaryGripperAction


class BinaryGripperAction(ActionTerm):
    cfg: BinaryGripperActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: BinaryGripperActionCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self._articulation = resolve_articulation(env, cfg.asset_name)
        matched = match_joints(cfg.joint_names, self._articulation.joint_names)
        if not matched:
            raise ValueError(
                f"BinaryGripperAction matched zero joints from patterns {cfg.joint_names!r}"
            )
        self._joint_indices = torch.tensor(matched, dtype=torch.long, device=self.device)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._target = torch.zeros(self.num_envs, len(matched), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        is_open = actions[:, 0] > float(self.cfg.threshold)
        pos = torch.where(
            is_open,
            torch.full_like(is_open, float(self.cfg.open_pos), dtype=self._target.dtype),
            torch.full_like(is_open, float(self.cfg.closed_pos), dtype=self._target.dtype),
        )
        self._target[:] = pos.unsqueeze(-1)

    def apply_actions(self) -> None:
        self._articulation.write_joint_targets_partial(self._joint_indices, self._target)
