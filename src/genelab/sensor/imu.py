"""Inertial Measurement Unit sensor.

Outputs orientation, body-frame projected gravity, and body-frame linear / angular
accelerations at a site rigidly attached to a robot link. Linear and angular accelerations
are computed by finite difference of the world-frame velocity buffers held internally —
the first step after every ``reset`` returns zero acceleration to avoid a spurious spike
from a stale ``prev`` value. The accelerometer convention is ``f_b = R^T (a_w - g_w)``
when ``gravity_bias=True`` (matches a real IMU at rest reporting ``+g`` along its up
axis); set ``gravity_bias=False`` to read pure kinematic acceleration instead.

The high-frequency jitter from ``(v - prev) / dt`` at the control rate is **not** filtered
internally — consumers can layer ``ObservationTermCfg.noise`` + ``scale`` on top, or apply
their own EMA. ``BodyVelocitySensor`` remains the velocimeter/gyro abstraction and is
expected to coexist with this sensor when both velocity and acceleration are needed.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_state
from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


_GRAVITY_MAGNITUDE = 9.81


@dataclass
class IMUSensorCfg(SensorCfg):
    """Configuration for ``IMUSensor``.

    ``offset`` is the site position in the link's local frame; it contributes both the
    lever-arm term to linear velocity (``ω × R · offset``) and, via its time-derivative,
    to linear acceleration. ``gravity_bias`` toggles real-IMU specific-force convention.
    ``bias_range_lin_acc`` and ``bias_range_ang_acc`` each sample a constant per-env bias
    on every ``reset(env_ids)`` to emulate sensor offset randomization.
    """

    link_name: str = ""
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_bias: bool = True
    bias_range_lin_acc: tuple[float, float] | None = None
    bias_range_ang_acc: tuple[float, float] | None = None

    def build(self) -> "IMUSensor":
        return IMUSensor(self)


@dataclass
class IMUData:
    orientation: torch.Tensor  # (B, 4) wxyz, link_quat_w
    projected_gravity_b: torch.Tensor  # (B, 3) unit gravity rotated into the link's body frame
    lin_acc_b: torch.Tensor  # (B, 3) linear acceleration at the site, body frame
    ang_acc_b: torch.Tensor  # (B, 3) angular acceleration of the link, body frame


class IMUSensor(Sensor[IMUData]):
    def __init__(self, cfg: IMUSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx: int = -1
        self._offset_local: torch.Tensor | None = None
        self._prev_lin_vel_w: torch.Tensor | None = None
        self._prev_ang_vel_w: torch.Tensor | None = None
        self._first_step: torch.Tensor | None = None
        self._lin_acc_b_buf: torch.Tensor | None = None
        self._ang_acc_b_buf: torch.Tensor | None = None
        self._bias_lin: torch.Tensor | None = None
        self._bias_ang: torch.Tensor | None = None

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        if not self._cfg_typed.link_name:
            raise ValueError(f"IMUSensorCfg(name={self._cfg.name!r}) requires link_name")
        link_names = entity_articulation(env, self._cfg.entity_name).link_names
        try:
            self._link_idx = link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={link_names!r}"
            ) from exc
        device = env.device
        self._offset_local = torch.tensor(
            self._cfg_typed.offset, dtype=torch.float32, device=device
        )
        zeros = lambda: torch.zeros(env.num_envs, 3, device=device)  # noqa: E731
        self._prev_lin_vel_w = zeros()
        self._prev_ang_vel_w = zeros()
        self._first_step = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        self._lin_acc_b_buf = zeros()
        self._ang_acc_b_buf = zeros()
        self._bias_lin = zeros()
        self._bias_ang = zeros()
        if self._cfg_typed.bias_range_lin_acc is not None:
            self._resample_bias(
                self._bias_lin,
                self._cfg_typed.bias_range_lin_acc,
                torch.arange(env.num_envs, device=device),
            )
        if self._cfg_typed.bias_range_ang_acc is not None:
            self._resample_bias(
                self._bias_ang,
                self._cfg_typed.bias_range_ang_acc,
                torch.arange(env.num_envs, device=device),
            )

    def update(self, dt: float) -> None:
        super().update(dt)
        assert (
            self._env is not None
            and self._offset_local is not None
            and self._prev_lin_vel_w is not None
            and self._prev_ang_vel_w is not None
            and self._first_step is not None
            and self._lin_acc_b_buf is not None
            and self._ang_acc_b_buf is not None
            and self._bias_lin is not None
            and self._bias_ang is not None
        )
        rs = entity_state(self._env, self._cfg.entity_name)
        link_quat = rs.link_quat_w[:, self._link_idx]
        link_ang_vel_w = rs.link_ang_vel_w[:, self._link_idx]
        link_lin_vel_w = rs.link_lin_vel_w[:, self._link_idx]
        # Site velocity in world frame: v_site = v_link + ω × (R · offset).
        rot = matrix_from_quat(link_quat)
        r_world = torch.matmul(rot, self._offset_local)
        v_site_w = link_lin_vel_w + torch.cross(link_ang_vel_w, r_world, dim=-1)

        first = self._first_step.unsqueeze(-1)
        lin_acc_w = torch.where(
            first, torch.zeros_like(v_site_w), (v_site_w - self._prev_lin_vel_w) / dt
        )
        ang_acc_w = torch.where(
            first, torch.zeros_like(link_ang_vel_w), (link_ang_vel_w - self._prev_ang_vel_w) / dt
        )

        lin_acc_b = quat_apply_inverse(link_quat, lin_acc_w)
        ang_acc_b = quat_apply_inverse(link_quat, ang_acc_w)
        if self._cfg_typed.gravity_bias:
            gravity_w = torch.zeros_like(lin_acc_w)
            gravity_w[..., 2] = -_GRAVITY_MAGNITUDE
            gravity_b = quat_apply_inverse(link_quat, gravity_w)
            lin_acc_b = lin_acc_b - gravity_b

        self._lin_acc_b_buf = lin_acc_b + self._bias_lin
        self._ang_acc_b_buf = ang_acc_b + self._bias_ang
        # Clone before storing so subsequent in-place writes to ``robot_state`` (which the
        # env does every control step) cannot retroactively alter our previous-tick buffer.
        self._prev_lin_vel_w = v_site_w.clone()
        self._prev_ang_vel_w = link_ang_vel_w.clone()
        self._first_step = torch.zeros_like(self._first_step)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if self._env is None or self._first_step is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        assert (
            self._prev_lin_vel_w is not None
            and self._prev_ang_vel_w is not None
            and self._lin_acc_b_buf is not None
            and self._ang_acc_b_buf is not None
            and self._bias_lin is not None
            and self._bias_ang is not None
        )
        self._prev_lin_vel_w[env_ids] = 0.0
        self._prev_ang_vel_w[env_ids] = 0.0
        self._lin_acc_b_buf[env_ids] = 0.0
        self._ang_acc_b_buf[env_ids] = 0.0
        self._first_step[env_ids] = True
        if self._cfg_typed.bias_range_lin_acc is not None:
            self._resample_bias(self._bias_lin, self._cfg_typed.bias_range_lin_acc, env_ids)
        if self._cfg_typed.bias_range_ang_acc is not None:
            self._resample_bias(self._bias_ang, self._cfg_typed.bias_range_ang_acc, env_ids)

    @staticmethod
    def _resample_bias(
        bias: torch.Tensor, bias_range: tuple[float, float], env_ids: torch.Tensor
    ) -> None:
        lo, hi = bias_range
        bias[env_ids] = torch.rand(env_ids.numel(), 3, device=bias.device) * (hi - lo) + lo

    def _compute_data(self) -> IMUData:
        assert (
            self._env is not None
            and self._lin_acc_b_buf is not None
            and self._ang_acc_b_buf is not None
        )
        rs = entity_state(self._env, self._cfg.entity_name)
        link_quat = rs.link_quat_w[:, self._link_idx]
        gravity_w = torch.zeros_like(rs.link_lin_vel_w[:, self._link_idx])
        gravity_w[..., 2] = -1.0
        projected_gravity_b = quat_apply_inverse(link_quat, gravity_w)
        return IMUData(
            orientation=link_quat,
            projected_gravity_b=projected_gravity_b,
            lin_acc_b=self._lin_acc_b_buf,
            ang_acc_b=self._ang_acc_b_buf,
        )
