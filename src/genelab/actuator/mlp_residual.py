"""Learned-residual actuator — DCMotor analytic base + an MLP correction (ROADMAP M2.5).

Models the gap between an idealized torque-speed motor curve and a real actuator by adding
a learned residual to the :class:`~genelab.actuator.dc_motor.DCMotorActuator` effort:

    effort = clamp(dcmotor_base(q, q̇, q*) + scale · net([q* − q, q̇]), ±effort_budget)

The residual network is a TorchScript module loaded from ``network_file`` (per the M2.5
design: GeneLab only *loads and runs* the net — training the weights lives in ``examples/``
or a downstream project). Its contract is single-step and per-joint:

* **input** ``(…, 2)`` — last dim is ``[target_pos − joint_pos, joint_vel]`` per joint,
* **output** ``(…, 1)`` — the residual torque per joint.

A standard MLP whose first layer is ``nn.Linear(2, …)`` satisfies this directly (the linear
acts on the trailing feature dim, so a ``(num_envs, num_joints, 2)`` batch maps to
``(num_envs, num_joints, 1)``). With no ``network_file`` the actuator degrades to a plain
``DCMotorActuator`` (zero residual).
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from genelab.actuator.actuator_base import ActuatorBaseCfg
from genelab.actuator.dc_motor import DCMotorActuator, DCMotorActuatorCfg


@dataclass
class MlpResidualActuatorCfg(DCMotorActuatorCfg):
    """Configuration for :class:`MlpResidualActuator`.

    Inherits the DCMotor fields (``velocity_limit`` required; ``saturation_effort`` falls
    back to ``effort_limit``). ``network_file`` is a TorchScript ``.pt`` whose forward maps
    ``[pos_error, joint_vel]`` (last dim) to a per-joint residual torque; ``None`` disables
    the residual (pure DCMotor). ``residual_scale`` scales the net output before it is added.
    """

    network_file: str | None = None
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = MlpResidualActuator
        # Validates velocity_limit (and leaves class_type as set above).
        super().__post_init__()


class MlpResidualActuator(DCMotorActuator):
    """DCMotor torque base plus a TorchScript MLP residual on ``[pos_error, joint_vel]``."""

    cfg: MlpResidualActuatorCfg  # type: ignore[assignment]
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
        if not isinstance(cfg, MlpResidualActuatorCfg):
            raise TypeError(
                f"MlpResidualActuator(name={name!r}) requires MlpResidualActuatorCfg, got "
                f"{type(cfg).__name__}"
            )
        self._residual_scale = float(cfg.residual_scale)
        self._net: Any = None
        if cfg.network_file is not None:
            net = torch.jit.load(cfg.network_file, map_location=device)
            net.eval()
            self._net = net

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = super().compute(joint_pos, joint_vel, target_pos, target_vel)
        if self._net is None:
            return base
        features = torch.stack([target_pos - joint_pos, joint_vel], dim=-1)  # (B, J, 2)
        with torch.no_grad():
            residual = self._net(features).reshape(base.shape)  # (B, J)
        effort = base + self._residual_scale * residual
        # Re-clamp the corrected effort to the hardware budget (effort_limit, else saturation).
        budget = self._effort_limit if self._effort_limit is not None else self._saturation_effort
        lim = budget.unsqueeze(0)
        return torch.clamp(effort, min=-lim, max=lim)
