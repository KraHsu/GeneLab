"""Frame transformer sensor.

Outputs the pose of one or more target frames relative to a source frame, using forward
kinematics over the link state already maintained by ``Articulation``. Both source and
target frames support a configurable local-frame offset (position + quaternion) so a sensor
can probe a specific point on a link (e.g. an end-effector grasp pose) without requiring the
MJCF to expose a dedicated body.

Stateless: ``_compute_data`` reads ``rs.link_pos`` / ``rs.link_quat_w`` directly each tick
and composes per-target offsets cached at ``bind`` time. No buffers, no integrators, no
finite-difference — ``update`` and ``reset`` only need the base-class cache invalidation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_state
from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class TargetFrameCfg:
    """A single target frame in a :class:`FrameTransformerSensorCfg`.

    ``link_name`` resolves to a link index at ``bind``. ``offset_pos`` / ``offset_quat`` apply
    a rigid offset in the link's local frame so the same link can host multiple targets at
    different attachment sites. ``name`` is the user-facing identifier — defaults to
    ``link_name`` when blank — and is exposed via ``FrameTransformerSensor.target_names``.
    """

    link_name: str = ""
    name: str = ""
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class FrameTransformerSensorCfg(SensorCfg):
    """Configuration for ``FrameTransformerSensor``.

    ``source_link_name`` is the reference frame the targets are expressed in. ``target_frames``
    is order-preserving — the output's ``N`` axis follows the cfg order so downstream
    observation terms can slice by position.
    """

    source_link_name: str = ""
    source_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_frames: tuple[TargetFrameCfg, ...] = field(default_factory=tuple)

    def build(self) -> "FrameTransformerSensor":
        return FrameTransformerSensor(self)


@dataclass
class FrameTransformerData:
    target_pos_w: torch.Tensor  # (B, N, 3) target positions in world frame
    target_quat_w: torch.Tensor  # (B, N, 4) target orientations in world frame
    target_pos_source: torch.Tensor  # (B, N, 3) target positions in source frame
    target_quat_source: torch.Tensor  # (B, N, 4) target orientations in source frame


class FrameTransformerSensor(Sensor[FrameTransformerData]):
    def __init__(self, cfg: FrameTransformerSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._source_idx: int = -1
        self._source_offset_pos: torch.Tensor | None = None
        self._source_offset_quat: torch.Tensor | None = None
        self._target_indices: torch.Tensor | None = None
        self._target_offsets_pos: torch.Tensor | None = None
        self._target_offsets_quat: torch.Tensor | None = None
        self._target_names: tuple[str, ...] = ()

    @property
    def target_names(self) -> tuple[str, ...]:
        return self._target_names

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        if not self._cfg_typed.source_link_name:
            raise ValueError(
                f"FrameTransformerSensorCfg(name={self._cfg.name!r}) requires source_link_name"
            )
        if not self._cfg_typed.target_frames:
            raise ValueError(
                f"FrameTransformerSensorCfg(name={self._cfg.name!r}) requires at least one target frame"
            )
        link_names = entity_articulation(env, self._cfg.entity_name).link_names
        try:
            self._source_idx = link_names.index(self._cfg_typed.source_link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: source link {self._cfg_typed.source_link_name!r} "
                f"not in link_names={link_names!r}"
            ) from exc
        unresolved: list[str] = [
            t.link_name for t in self._cfg_typed.target_frames if t.link_name not in link_names
        ]
        if unresolved:
            raise ValueError(
                f"sensor {self._cfg.name!r}: target link(s) {unresolved!r} not in "
                f"link_names={link_names!r}"
            )
        device = env.device
        self._source_offset_pos = torch.tensor(
            self._cfg_typed.source_offset_pos, dtype=torch.float32, device=device
        )
        self._source_offset_quat = torch.tensor(
            self._cfg_typed.source_offset_quat, dtype=torch.float32, device=device
        )
        self._target_indices = torch.tensor(
            [link_names.index(t.link_name) for t in self._cfg_typed.target_frames],
            dtype=torch.long,
            device=device,
        )
        self._target_offsets_pos = torch.tensor(
            [list(t.offset_pos) for t in self._cfg_typed.target_frames],
            dtype=torch.float32,
            device=device,
        )
        self._target_offsets_quat = torch.tensor(
            [list(t.offset_quat) for t in self._cfg_typed.target_frames],
            dtype=torch.float32,
            device=device,
        )
        self._target_names = tuple(
            t.name if t.name else t.link_name for t in self._cfg_typed.target_frames
        )

    def _compute_data(self) -> FrameTransformerData:
        assert (
            self._env is not None
            and self._source_offset_pos is not None
            and self._source_offset_quat is not None
            and self._target_indices is not None
            and self._target_offsets_pos is not None
            and self._target_offsets_quat is not None
        )
        rs = entity_state(self._env, self._cfg.entity_name)
        b = self._env.num_envs
        n = self._target_indices.shape[0]

        # Source pose in world frame, then composed with the source-local offset.
        src_link_pos = rs.link_pos[:, self._source_idx]  # (B, 3)
        src_link_quat = rs.link_quat_w[:, self._source_idx]  # (B, 4)
        src_offset_pos_b = self._source_offset_pos.unsqueeze(0).expand(b, 3).contiguous()
        src_offset_quat_b = self._source_offset_quat.unsqueeze(0).expand(b, 4).contiguous()
        source_pos_w = src_link_pos + quat_apply(src_link_quat, src_offset_pos_b)
        source_quat_w = quat_mul(src_link_quat, src_offset_quat_b)

        # Per-target link pose, then composed with the target-local offset.
        tgt_link_pos = rs.link_pos[:, self._target_indices]  # (B, N, 3)
        tgt_link_quat = rs.link_quat_w[:, self._target_indices]  # (B, N, 4)
        tgt_offset_pos_b = self._target_offsets_pos.unsqueeze(0).expand(b, n, 3).contiguous()
        tgt_offset_quat_b = self._target_offsets_quat.unsqueeze(0).expand(b, n, 4).contiguous()
        target_pos_w = tgt_link_pos + quat_apply(tgt_link_quat, tgt_offset_pos_b)
        target_quat_w = quat_mul(tgt_link_quat, tgt_offset_quat_b)

        # Express each target in the source frame: t_source = inv(source) @ t_target.
        source_pos_expanded = source_pos_w.unsqueeze(1).expand(b, n, 3).contiguous()
        source_quat_expanded = source_quat_w.unsqueeze(1).expand(b, n, 4).contiguous()
        target_pos_source, target_quat_source = subtract_frame_transforms(
            source_pos_expanded, source_quat_expanded, target_pos_w, target_quat_w
        )
        return FrameTransformerData(
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            target_pos_source=target_pos_source,
            target_quat_source=target_quat_source,
        )
