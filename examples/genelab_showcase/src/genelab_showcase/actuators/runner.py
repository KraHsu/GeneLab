"""Actuator-showcase runner: drive joint1 with a sine; print target vs actual."""

import math
from typing import TYPE_CHECKING

import torch

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class ActuatorShowcaseRunner(ShowcaseRunner):
    """Sine joint-1 target through an IdealPDActuator arm; dumps tracking error.

    Output dump (``logs/showcase/actuators/tracking.log``) every 20 steps:

    ::

        step=NNNN  target=A.AAA  actual=B.BBB  error=C.CCC  joint1_vel=D.DDD
    """

    task_slug = "actuators"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=20)
        self._joint1_idx: int | None = None

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        if self._joint1_idx is None:
            self._joint1_idx = env.articulations["robot"].joint_names.index("joint1")
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        phase = 2.0 * math.pi * step / 200.0
        action[:, self._joint1_idx] = math.sin(phase) * 0.8
        return action

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if self._joint1_idx is None:
            return
        root = self.log_root()
        rs = env.articulations["robot"].data
        joint_pos = float(rs.joint_pos[0, self._joint1_idx])
        joint_vel = float(rs.joint_vel[0, self._joint1_idx])
        default = float(env.articulations["robot"].default_joint_pos[self._joint1_idx])
        phase = 2.0 * math.pi * step / 200.0
        # Reconstruct the absolute target: default + action_scale * raw_action.
        # action_scale 0.5 lines up with the IdealPDActuator group in env_cfg.
        action_scale = 0.5
        raw = math.sin(phase) * 0.8
        target = default + action_scale * raw
        error = target - joint_pos
        with (root / "tracking.log").open("a", encoding="utf-8") as fh:
            fh.write(
                f"step={step:04d}  target={target:.4f}  actual={joint_pos:.4f}  "
                f"error={error:+.4f}  joint1_vel={joint_vel:+.4f}\n"
            )


class MlpResidualActuatorShowcaseRunner(ActuatorShowcaseRunner):
    """Same scripted joint-1 sweep + tracking dump, but the arm runs on an
    ``MlpResidualActuator`` (DC-motor base + TorchScript residual). Distinct log slug
    so its ``tracking.log`` does not collide with the IdealPD actuator showcase."""

    task_slug = "mlp_residual_actuator"
