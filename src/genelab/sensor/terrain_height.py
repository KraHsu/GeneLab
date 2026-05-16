"""Terrain height sensor: per-frame vertical clearance against ground.

Two modes, distinguished by which cfg field is populated:

* **Single-frame (legacy)** — set ``link_name`` (singular). Output shape is
  ``(B, num_rays)``: one height per ray, all anchored to a single link. Matches
  the pre-P3 API; existing call sites keep working.
* **Multi-frame** — set ``link_names`` (plural tuple) and pick a ``reduction``.
  Output shape is ``(B, F)`` after reducing the per-frame ray fan to a single
  number, or ``(B, F, num_rays)`` with ``reduction="none"``. mjlab parity for
  ``foot_height_scan`` — one sensor scans both feet at once and gives back one
  clearance per foot.

Composes a ``RayCastSensor`` per frame; lifecycle (``update`` / ``reset``) fans
out to all inners. The default flat-plane backend gives ``link_z - ground_z``
per ray; for non-flat terrain the inner ``RayCastSensor`` walks a heightfield
via ``_intersect_world_rays``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from genelab.sensor.ray_cast import (
    GridPattern,
    HemispherePattern,
    RayCastSensor,
    RayCastSensorCfg,
    RingPattern,
)
from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


Reduction = Literal["min", "max", "mean", "none"]


@dataclass
class TerrainHeightSensorCfg(SensorCfg):
    # Single-frame legacy entrypoint. Set this *or* ``link_names``, not both.
    link_name: str = ""
    # Multi-frame entrypoint (mjlab parity); one frame per link, reduction applied
    # per-frame across the pattern.
    link_names: tuple[str, ...] = ()
    pattern: GridPattern | RingPattern | HemispherePattern = field(default_factory=GridPattern)
    reduction: Reduction = "min"
    attach_yaw_only: bool = True
    max_distance: float = 10.0
    ground_height: float = 0.0
    link_offsets: tuple[tuple[float, float, float], ...] | None = None
    """Per-link local-frame offsets for the ray-pattern origin (mjlab site parity).

    When set in multi-frame mode, must have the same length as ``link_names``;
    each entry is the site position in the corresponding link's body frame.
    G1's foot-height scan uses ``((0.04, 0, -0.037), (0.04, 0, -0.037))`` to
    sample under the foot site rather than the ankle_roll_link origin —
    important on flat ground where the ~0.037 m site z-offset matters next to
    a 0.1 m target swing height.

    In single-frame mode (``link_name`` set), pass a length-1 tuple. ``None``
    keeps all offsets at zero (pre-parity behaviour).
    """

    def __post_init__(self) -> None:
        has_single = bool(self.link_name)
        has_multi = bool(self.link_names)
        if has_single and has_multi:
            raise ValueError(
                f"TerrainHeightSensorCfg(name={self.name!r}): set either link_name "
                f"(single-frame legacy) or link_names (multi-frame), not both"
            )
        if not has_single and not has_multi:
            raise ValueError(
                f"TerrainHeightSensorCfg(name={self.name!r}): set either link_name or link_names"
            )
        if self.link_offsets is not None:
            expected = 1 if has_single else len(self.link_names)
            if len(self.link_offsets) != expected:
                raise ValueError(
                    f"TerrainHeightSensorCfg(name={self.name!r}): link_offsets length "
                    f"({len(self.link_offsets)}) must match frame count ({expected})"
                )

    def build(self) -> "TerrainHeightSensor":
        return TerrainHeightSensor(self)


class TerrainHeightSensor(Sensor[torch.Tensor]):
    """Per-frame vertical clearance above the ground/terrain.

    Output shape depends on cfg mode (see :class:`TerrainHeightSensorCfg`):

    * Single-frame: ``(B, num_rays)`` — every ray's ``frame_z - hit_z``.
    * Multi-frame, ``reduction="none"``: ``(B, F, num_rays)``.
    * Multi-frame, ``reduction="min"/"max"/"mean"``: ``(B, F)`` — one height per
      frame after reducing across the ray fan.
    """

    def __init__(self, cfg: TerrainHeightSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        # Normalise the frame spec to a (possibly singleton) tuple and remember which
        # branch was active so ``_compute_data`` can preserve the legacy output shape.
        self._legacy_single_frame = bool(cfg.link_name)
        frame_link_names: tuple[str, ...] = (
            (cfg.link_name,) if self._legacy_single_frame else cfg.link_names
        )
        # Per-frame local offset: pass through to the inner RayCastSensor so the ray
        # fan anchors at the configured site rather than the link origin. ``None``
        # means "no offset" — produces an all-zero per-frame value.
        offsets: tuple[tuple[float, float, float], ...] = (
            cfg.link_offsets
            if cfg.link_offsets is not None
            else tuple((0.0, 0.0, 0.0) for _ in frame_link_names)
        )

        # One inner RayCastSensor per frame. The link count here is small (typically
        # two feet); the per-frame instantiation cost is amortised against many steps.
        self._inners: list[RayCastSensor] = [
            RayCastSensor(
                RayCastSensorCfg(
                    name=f"{cfg.name}__{frame_name}",
                    link_name=frame_name,
                    pattern=cfg.pattern,
                    attach_yaw_only=cfg.attach_yaw_only,
                    max_distance=cfg.max_distance,
                    ground_height=cfg.ground_height,
                    link_offset=offset,
                )
            )
            for frame_name, offset in zip(frame_link_names, offsets, strict=True)
        ]

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        for inner in self._inners:
            inner.bind(env)

    def update(self, dt: float) -> None:
        super().update(dt)
        for inner in self._inners:
            inner.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        for inner in self._inners:
            inner.reset(env_ids)

    def _compute_data(self) -> torch.Tensor:
        # Per-frame per-ray clearance: positive when the link is above the hit point.
        per_frame_heights = [
            inner.data.ray_starts_w[..., 2] - inner.data.hit_pos_w[..., 2] for inner in self._inners
        ]

        if self._legacy_single_frame:
            # Preserve the pre-P3 output shape (B, num_rays) so call sites that still
            # pass ``link_name`` keep their tensor layout.
            return per_frame_heights[0]

        stacked = torch.stack(per_frame_heights, dim=1)  # (B, F, num_rays)
        reduction = self._cfg_typed.reduction
        if reduction == "none":
            return stacked
        if reduction == "min":
            return stacked.min(dim=-1).values
        if reduction == "max":
            return stacked.max(dim=-1).values
        if reduction == "mean":
            return stacked.mean(dim=-1)
        raise ValueError(f"Unknown reduction: {reduction!r}")
