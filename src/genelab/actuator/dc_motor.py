"""DC-motor actuator — IdealPD with velocity-dependent torque saturation.

Adds a linear de-rating curve on top of :class:`IdealPDActuator`:

* ``tau_max(q_dot) = saturation_effort * clip(1 - |q_dot| / velocity_limit, 0, 1)``

The de-rating is applied **only in the driving direction** (``sign(tau_pd) == sign(q_dot)``);
reverse braking torque is left at the full :attr:`effort_limit`. This is the Isaac Lab
``DCMotor`` semantics — modelling the back-EMF saturation of a permanent-magnet DC motor
without penalising regenerative braking.

``saturation_effort`` defaults to :attr:`effort_limit` when unset.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from genelab.actuator.actuator_base import ActuatorBaseCfg
from genelab.actuator.ideal_pd import IdealPDActuator


@dataclass
class DCMotorActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`DCMotorActuator`.

    ``velocity_limit`` is required (used as the no-load speed in the de-rating curve).
    ``saturation_effort`` falls back to ``effort_limit`` when ``None``.
    """

    saturation_effort: float | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = DCMotorActuator
        if self.velocity_limit is None:
            raise ValueError(
                "DCMotorActuatorCfg requires velocity_limit (used as no-load speed in the "
                "torque-speed de-rating curve)"
            )


class DCMotorActuator(IdealPDActuator):
    """IdealPD + linear torque-speed de-rating on the driving direction only."""

    cfg: DCMotorActuatorCfg  # type: ignore[assignment]
    channel: Literal["implicit_pd", "force"] = "force"

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        *,
        name: str,
        joint_ids: torch.Tensor,
        dof_ids: torch.Tensor,
        joint_names: list[str],
        num_envs: int,
        device: str,
    ) -> None:
        super().__init__(
            cfg,
            name=name,
            joint_ids=joint_ids,
            dof_ids=dof_ids,
            joint_names=joint_names,
            num_envs=num_envs,
            device=device,
        )
        if not isinstance(cfg, DCMotorActuatorCfg):
            raise TypeError(
                f"DCMotorActuator(name={name!r}) requires DCMotorActuatorCfg, got "
                f"{type(cfg).__name__}"
            )
        sat = cfg.saturation_effort if cfg.saturation_effort is not None else cfg.effort_limit
        if sat is None:
            raise ValueError(
                f"DCMotorActuator(name={name!r}): needs effort_limit or saturation_effort"
            )
        self._saturation_effort = torch.full((self._num_joints,), float(sat), device=self._device)

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kp_scale, kv_scale = self.gain_scales(joint_pos.shape[0])
        kp = self._stiffness.unsqueeze(0) * kp_scale
        kv = self._damping.unsqueeze(0) * kv_scale
        tau_pd = kp * (target_pos - joint_pos) - kv * joint_vel

        # velocity_limit is mandatory in DCMotorActuatorCfg.__post_init__; the assert is a
        # type-narrowing hint for pyright.
        assert self._velocity_limit is not None
        v_max = self._velocity_limit.unsqueeze(0)
        sat = self._saturation_effort.unsqueeze(0)
        derate = torch.clamp(1.0 - joint_vel.abs() / v_max, min=0.0, max=1.0)
        tau_max_drive = sat * derate
        brake_limit = self._effort_limit.unsqueeze(0) if self._effort_limit is not None else sat

        # Asymmetric clamp keyed by joint_vel sign: the "driving" direction (positive torque
        # when q_dot >= 0, negative torque when q_dot < 0) is bounded by the de-rated limit;
        # the opposite "braking" direction keeps the full effort budget.
        v_nonneg = joint_vel >= 0
        upper = torch.where(v_nonneg, tau_max_drive, brake_limit)
        lower = torch.where(v_nonneg, -brake_limit, -tau_max_drive)
        return torch.clamp(tau_pd, min=lower, max=upper)
