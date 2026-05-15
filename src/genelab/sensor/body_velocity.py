"""Velocity sensor at a site rigidly attached to a robot link.

Mirrors MuJoCo's ``<velocimeter site=...>`` and ``<gyro site=...>``: the value is reported in the
link's body frame, with the velocimeter accounting for the lever arm from link origin to site.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class BodyVelocitySensorCfg(SensorCfg):
    """Configuration for ``BodyVelocitySensor``.

    ``offset`` is the site position in the link's local frame; ignored when ``measure="ang_vel"``
    since angular velocity does not depend on the lever arm. ``bias_range``, when set, samples a
    constant per-env bias on every ``reset(env_ids)`` to emulate sensor offset randomization.
    """

    link_name: str = ""
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    measure: Literal["lin_vel", "ang_vel"] = "lin_vel"
    bias_range: tuple[float, float] | None = None

    def build(self) -> "BodyVelocitySensor":
        return BodyVelocitySensor(self)


class BodyVelocitySensor(Sensor[torch.Tensor]):
    def __init__(self, cfg: BodyVelocitySensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx: int = -1
        self._offset_local: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        if not self._cfg_typed.link_name:
            raise ValueError(f"BodyVelocitySensorCfg(name={self._cfg.name!r}) requires link_name")
        try:
            self._link_idx = env.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not found in "
                f"env.link_names={env.link_names!r}"
            ) from exc
        self._offset_local = torch.tensor(
            self._cfg_typed.offset, dtype=torch.float32, device=env.device
        )
        self._bias = torch.zeros(env.num_envs, 3, device=env.device)
        if self._cfg_typed.bias_range is not None:
            self._resample_bias(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if self._cfg_typed.bias_range is None or self._bias is None or self._env is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._resample_bias(env_ids)

    def _resample_bias(self, env_ids: torch.Tensor) -> None:
        assert self._bias is not None and self._cfg_typed.bias_range is not None
        lo, hi = self._cfg_typed.bias_range
        self._bias[env_ids] = (
            torch.rand(env_ids.numel(), 3, device=self._bias.device) * (hi - lo) + lo
        )

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None and self._offset_local is not None and self._bias is not None
        rs = self._env.robot_state
        link_quat = rs.link_quat_w[:, self._link_idx]
        link_ang_vel_w = rs.link_ang_vel_w[:, self._link_idx]
        if self._cfg_typed.measure == "ang_vel":
            return quat_apply_inverse(link_quat, link_ang_vel_w) + self._bias
        link_lin_vel_w = rs.link_lin_vel_w[:, self._link_idx]
        r_world = quat_apply(link_quat, self._offset_local.expand_as(link_lin_vel_w))
        v_site_w = link_lin_vel_w + torch.cross(link_ang_vel_w, r_world, dim=-1)
        return quat_apply_inverse(link_quat, v_site_w) + self._bias
