"""Single-dim continuous gripper action term — integrates a scalar policy
output into a finger-width target, mirroring panda-gym's gripper control.

panda-gym's ``Panda`` advances the gripper by ``Δwidth = action · 0.2`` per
control step (``width`` being the sum of the two finger positions) and commands
each finger to ``width / 2``. This term reproduces that surface: the policy
emits one scalar per env, the term reads the *sensed* mean finger position,
adds ``action · speed`` and clamps the result to ``[closed_pos, open_pos]``
before broadcasting it across the matched finger joints.

Companions :class:`DifferentialIKAction` to reproduce panda-gym's 4-DoF
``(dx, dy, dz, gripper)`` Cartesian action space. Prefer this over
:class:`BinaryGripperAction` when the policy is Gaussian — a binary mapping
flips the gripper open/closed on the sign of a noisy action every step, which
makes a stable grasp statistically unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.action_manager import ActionTerm
from genelab.mdp._helpers import resolve_robot_state
from genelab.mdp.actions._gripper_base import GripperActionBase, GripperActionCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class ContinuousGripperActionCfg(GripperActionCfg):
    """One-dim continuous gripper action.

    The policy emits a single scalar per env; the term moves the matched finger
    joints by ``action · speed`` per control step, integrating on the *sensed*
    mean finger position and clamping the result to ``[closed_pos, open_pos]``.

    Defaults match the Franka Panda's ``finger_joint*`` range (``0.0`` closed →
    ``0.04`` open). ``speed`` is the per-finger displacement at ``|action| = 1``;
    the default ``0.1`` reproduces panda-gym's ``Δwidth = action · 0.2`` (width
    being the two-finger sum, so half that per finger). Use one regex per finger
    group via ``joint_names``.
    """

    speed: float = 0.1
    clip: tuple[float, float] | None = (-1.0, 1.0)
    class_type: type[ActionTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ContinuousGripperAction


class ContinuousGripperAction(GripperActionBase):
    cfg: ContinuousGripperActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: ContinuousGripperActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._robot_state = resolve_robot_state(env, cfg.asset_name)

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.clip is not None:
            lo, hi = self.cfg.clip
            actions = actions.clamp(min=lo, max=hi)
        self._raw[:] = actions
        # Integrate on the *sensed* mean finger position — panda-gym reads
        # ``get_fingers_width()`` each step rather than the last command, so a
        # blocked gripper (cube held / joint at a limit) does not let the target
        # drift away from the physical state.
        sensed = self._robot_state.joint_pos.index_select(1, self._joint_indices)
        width = sensed.mean(dim=1, keepdim=True)  # (B, 1)
        target = (width + actions[:, :1] * float(self.cfg.speed)).clamp(
            min=float(self.cfg.closed_pos), max=float(self.cfg.open_pos)
        )
        self._target[:] = target  # broadcast (B, 1) -> (B, n_fingers)
