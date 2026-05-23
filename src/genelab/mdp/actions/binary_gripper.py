"""Single-dim binary gripper action term — maps a scalar policy output to an
``{open, closed}`` finger-joint target broadcast across the matched joints.

Companions :class:`DifferentialIKAction` to reproduce panda-gym's 4-DoF
``(dx, dy, dz, gripper)`` Cartesian action space without entangling the IK
solver with the gripper degree of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from genelab.managers.action_manager import ActionTerm
from genelab.mdp.actions._gripper_base import GripperActionBase, GripperActionCfg


@dataclass
class BinaryGripperActionCfg(GripperActionCfg):
    """One-dim binary gripper action.

    The policy emits a single scalar per env; values strictly greater than
    ``threshold`` snap the matched finger joints to ``open_pos``, the rest to
    ``closed_pos``. Defaults match the Franka Panda's ``finger_joint*`` range
    (``0.0`` closed → ``0.04`` open). Use one regex per finger group via
    ``joint_names``.
    """

    threshold: float = 0.0
    class_type: type[ActionTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = BinaryGripperAction


class BinaryGripperAction(GripperActionBase):
    cfg: BinaryGripperActionCfg  # type: ignore[assignment]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        is_open = actions[:, 0] > float(self.cfg.threshold)
        pos = torch.where(
            is_open,
            torch.full_like(is_open, float(self.cfg.open_pos), dtype=self._target.dtype),
            torch.full_like(is_open, float(self.cfg.closed_pos), dtype=self._target.dtype),
        )
        self._target[:] = pos.unsqueeze(-1)
