"""Joint-position action term: ``target = default + scale * raw_action`` (PD-controlled)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class JointPositionActionCfg(ActionTermCfg):
    """Action term that pushes joint position targets through Genesis PD control.

    The configured ``joint_names`` are regex patterns matched against the env's joint names.
    Per-joint ``scale`` may be a single float or a dict (regex -> scale).
    """

    joint_names: tuple[str, ...] = (".*",)
    scale: float | dict[str, float] = 1.0
    use_default_offset: bool = True
    class_type: type[ActionTerm] | None = None  # filled by __post_init__

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = JointPositionAction


class JointPositionAction(ActionTerm):
    cfg: JointPositionActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: JointPositionActionCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        import re

        joint_names = env.joint_names
        matched: list[int] = []
        for pat in cfg.joint_names:
            try:
                regex = re.compile(pat)
            except re.error:
                regex = re.compile(re.escape(pat))
            for i, name in enumerate(joint_names):
                if regex.fullmatch(name) or regex.search(name):
                    if i not in matched:
                        matched.append(i)
        if not matched:
            raise ValueError(
                f"JointPositionAction matched zero joints from patterns {cfg.joint_names}"
            )
        self._joint_indices = torch.tensor(matched, dtype=torch.long, device=self.device)

        if isinstance(cfg.scale, dict):
            scale = env.articulation.build_per_joint_tensor(cfg.scale, default=1.0)
            self._scale = scale[self._joint_indices]
        else:
            self._scale = torch.full((len(matched),), float(cfg.scale), device=self.device)

        default = env.default_joint_pos[self._joint_indices]
        self._default = default if cfg.use_default_offset else torch.zeros_like(default)

        self._raw = torch.zeros(self.num_envs, len(matched), device=self.device)
        self._target = torch.zeros_like(self._raw)
        self._dim = len(matched)

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        self._target = self._default.unsqueeze(0) + self._scale.unsqueeze(0) * actions

    def apply_actions(self) -> None:
        control = getattr(self._env.robot, "control_dofs_position", None)
        if control is None:
            control = getattr(self._env.robot, "set_dofs_position_target", None)
        if control is None:
            return
        # Use the articulation-resolved actuated DoF indices (global), filtered by joint mask.
        dof_indices = self._env.articulation.actuated_dof_idx.index_select(0, self._joint_indices)
        try:
            control(self._target, dof_indices)
        except TypeError:
            control(self._target)
