"""Shared construction + IO for the single-dim gripper action terms.

:class:`BinaryGripperAction` and :class:`ContinuousGripperAction` resolve the same
finger joints, allocate the same buffers, and push targets the same way — they
differ only in how a scalar policy output maps to a finger-width target
(:meth:`process_actions`). This base owns everything they share so that mapping is
the only thing each subclass defines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm, ActionTermCfg
from genelab.mdp._helpers import resolve_articulation
from genelab.mdp.actions._joint_match import match_joints

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class GripperActionCfg(ActionTermCfg):
    """Config fields shared by the gripper action terms.

    Defaults match the Franka Panda's ``finger_joint*`` range (``0.0`` closed →
    ``0.04`` open). Use one regex per finger group via ``joint_names``.
    """

    joint_names: tuple[str, ...] = (".*",)
    open_pos: float = 0.04
    closed_pos: float = 0.0


class GripperActionBase(ActionTerm):
    """Resolves the finger joints + buffers and pushes targets to the articulation.

    Subclasses implement only :meth:`process_actions` (the binary snap vs. the
    continuous integrate). ``action_dim``, ``raw_actions`` and ``apply_actions``
    are identical for both and live here.
    """

    cfg: GripperActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: GripperActionCfg, env: EnvContext) -> None:
        super().__init__(cfg, env)
        self._articulation = resolve_articulation(env, cfg.asset_name)
        matched = match_joints(cfg.joint_names, self._articulation.joint_names)
        if not matched:
            raise ValueError(
                f"{type(self).__name__} matched zero joints from patterns {cfg.joint_names!r}"
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

    def apply_actions(self) -> None:
        self._articulation.write_joint_targets_partial(self._joint_indices, self._target)
