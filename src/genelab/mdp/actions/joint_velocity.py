"""Joint-velocity action term: ``target_velocity = scale * raw_action``.

For continuously-rotating joints — e.g. the Unitree Go2-W wheels — a position target is
meaningless, so the policy commands a joint *velocity* instead. Targets are routed through
:meth:`Articulation.write_joint_velocity_targets_partial`, which drives the matched joints'
velocity-channel actuators (:class:`~genelab.actuator.ImplicitVelocityActuator`) via Genesis
``control_dofs_velocity``. Compose it with a :class:`~genelab.mdp.actions.joint_position.JointPositionAction`
on the remaining joints (position legs + velocity wheels).

Unlike the position term there is no default offset (wheels spin from zero) and no
encoder-bias correction (that models a position-encoder zero-point, not a wheel velocity).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm, ActionTermCfg
from genelab.mdp._helpers import resolve_articulation

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class JointVelocityActionCfg(ActionTermCfg):
    """Action term that pushes joint velocity targets through velocity-channel actuators.

    ``joint_names`` are regex patterns matched against the env's joint names. ``scale`` maps
    the raw action to a target velocity (rad/s):

    * ``None`` — inherit per-joint scale from :attr:`Articulation.action_scale_tensor`.
    * ``float`` — single multiplier (e.g. the max wheel speed) applied to every matched joint.
    * ``dict[str, float]`` — per-regex override against joint names.
    """

    joint_names: tuple[str, ...] = (".*",)
    scale: float | dict[str, float] | None = None
    class_type: type[ActionTerm] | None = None  # filled by __post_init__

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = JointVelocityAction


class JointVelocityAction(ActionTerm):
    cfg: JointVelocityActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: JointVelocityActionCfg, env: "EnvContext") -> None:
        super().__init__(cfg, env)
        import re

        self._articulation = resolve_articulation(env, cfg.asset_name)
        joint_names = self._articulation.joint_names
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
                f"JointVelocityAction matched zero joints from patterns {cfg.joint_names}"
            )
        self._joint_indices = torch.tensor(matched, dtype=torch.long, device=self.device)

        if cfg.scale is None:
            self._scale = self._articulation.action_scale_tensor.index_select(
                0, self._joint_indices
            )
        elif isinstance(cfg.scale, dict):
            scale = self._articulation.build_per_joint_tensor(cfg.scale, default=1.0)
            self._scale = scale[self._joint_indices]
        else:
            self._scale = torch.full((len(matched),), float(cfg.scale), device=self.device)

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
        self._target = self._scale.unsqueeze(0) * actions

    def apply_actions(self) -> None:
        self._articulation.write_joint_velocity_targets_partial(self._joint_indices, self._target)
