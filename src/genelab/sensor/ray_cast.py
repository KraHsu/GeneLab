"""Ray-cast sensor with a configurable grid pattern, attached to a robot link.

Two intersection backends:

* **Flat plane** (default when ``scene.terrain is None``): every ray hits the horizontal
  plane at ``ground_height``. Closed-form intersection.
* **Height field** (when ``scene.terrain`` is a :class:`TerrainImporter`): bilinearly
  samples ``terrain.heightfield_tensor`` at each ray's ``(x, y)`` origin and uses the
  sampled elevation as the hit plane height. Accurate for vertical / near-vertical rays
  (the dominant case for height-scan grids); subclasses can override
  ``_intersect_world_rays`` for non-trivial BVH ray-tracing.

Pattern shape mirrors mjlab's ``GridPatternCfg``; for now only a 2D grid is provided.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.terrains import TerrainImporter


@dataclass
class GridPattern:
    """2D grid of ray origins centred at the sensor frame, all pointing along ``direction``."""

    resolution: float = 0.1
    size: tuple[float, float] = (1.6, 1.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def num_rays(self) -> int:
        nx = int(round(self.size[0] / self.resolution)) + 1
        ny = int(round(self.size[1] / self.resolution)) + 1
        return nx * ny

    def generate(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(starts_local, dirs_local)``, each shape ``(M, 3)``."""
        nx = int(round(self.size[0] / self.resolution)) + 1
        ny = int(round(self.size[1] / self.resolution)) + 1
        xs = torch.linspace(-self.size[0] / 2, self.size[0] / 2, nx, device=device)
        ys = torch.linspace(-self.size[1] / 2, self.size[1] / 2, ny, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        starts = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), torch.zeros_like(xx).reshape(-1)], dim=-1
        )
        dir_tensor = torch.tensor(self.direction, dtype=torch.float32, device=device)
        # Normalise so distances match world units even if direction wasn't unit length.
        dir_tensor = dir_tensor / dir_tensor.norm().clamp_min(1e-9)
        dirs = dir_tensor.unsqueeze(0).expand(starts.shape[0], -1).contiguous()
        return starts, dirs


@dataclass
class RayCastData:
    distances: torch.Tensor  # (B, M) — clamped to max_distance; equals max_distance on a miss
    hit_pos_w: torch.Tensor  # (B, M, 3) — world-frame intersection points
    normals_w: torch.Tensor  # (B, M, 3) — surface normals at hit
    ray_starts_w: torch.Tensor  # (B, M, 3) — world-frame ray origins
    ray_dirs_w: torch.Tensor  # (B, M, 3) — world-frame ray directions


@dataclass
class RayCastSensorCfg(SensorCfg):
    """Configuration for ``RayCastSensor``.

    ``link_name`` anchors the pattern. ``attach_yaw_only`` keeps the grid axis-aligned to the
    horizon when the link rolls / pitches (typical for terrain height scans on a torso link);
    set ``False`` to follow the full link orientation. ``ground_height`` is the z of the flat
    plane the default backend intersects against; subclasses ignore it.
    """

    link_name: str = ""
    pattern: GridPattern = field(default_factory=GridPattern)
    attach_yaw_only: bool = True
    max_distance: float = 10.0
    ground_height: float = 0.0

    def build(self) -> "RayCastSensor":
        return RayCastSensor(self)


class RayCastSensor(Sensor[RayCastData]):
    def __init__(self, cfg: RayCastSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx: int = -1
        self._ray_starts_local: torch.Tensor | None = None
        self._ray_dirs_local: torch.Tensor | None = None

    @property
    def num_rays(self) -> int:
        return self._cfg_typed.pattern.num_rays()

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        if not self._cfg_typed.link_name:
            raise ValueError(f"RayCastSensorCfg(name={self._cfg.name!r}) requires link_name")
        try:
            self._link_idx = env.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"env.link_names={env.link_names!r}"
            ) from exc
        self._ray_starts_local, self._ray_dirs_local = self._cfg_typed.pattern.generate(env.device)

    def _compute_data(self) -> RayCastData:
        assert (
            self._env is not None
            and self._ray_starts_local is not None
            and self._ray_dirs_local is not None
        )
        rs = self._env.robot_state
        link_pos = rs.link_pos[:, self._link_idx]
        link_quat = rs.link_quat_w[:, self._link_idx]
        rot_q = yaw_quat(link_quat) if self._cfg_typed.attach_yaw_only else link_quat
        # Project the local pattern into world coordinates per env.
        b = link_pos.shape[0]
        m = self._ray_starts_local.shape[0]
        q_expanded = rot_q.unsqueeze(1).expand(b, m, 4).contiguous()
        starts_local_b = self._ray_starts_local.unsqueeze(0).expand(b, m, 3).contiguous()
        dirs_local_b = self._ray_dirs_local.unsqueeze(0).expand(b, m, 3).contiguous()
        starts_w = quat_apply(q_expanded, starts_local_b) + link_pos.unsqueeze(1)
        dirs_w = quat_apply(q_expanded, dirs_local_b)
        distances, hit_pos_w, normals_w = self._intersect_world_rays(starts_w, dirs_w)
        return RayCastData(
            distances=distances,
            hit_pos_w=hit_pos_w,
            normals_w=normals_w,
            ray_starts_w=starts_w,
            ray_dirs_w=dirs_w,
        )

    def _intersect_world_rays(
        self, starts_w: torch.Tensor, dirs_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Intersect rays against the scene terrain or a horizontal default plane.

        Subclasses may override to plug in BVH or other custom backends.
        Returns ``(distances, hit_pos_w, normals_w)`` each shaped ``(B, M, ...)``.
        """
        assert self._env is not None
        terrain = self._scene_terrain()
        if terrain is None:
            ground_z = torch.full_like(starts_w[..., 2], self._cfg_typed.ground_height)
        else:
            ground_z = self._sample_heightfield(starts_w[..., :2], terrain)
        return self._ray_plane_hit(starts_w, dirs_w, ground_z)

    def _ray_plane_hit(
        self,
        starts_w: torch.Tensor,
        dirs_w: torch.Tensor,
        ground_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form intersection between rays and the (possibly per-ray) plane ``z = ground_z``.

        ``ground_z`` must broadcast against ``starts_w[..., 2]`` (so a scalar, a ``(B,)``
        per-env tensor, or a ``(B, M)`` per-ray tensor all work).
        """
        max_dist = self._cfg_typed.max_distance
        dir_z = dirs_w[..., 2]
        # Avoid division-by-zero / wrong-side hits (ray going up or parallel) by masking them
        # to a "miss" with distance = max_dist and hit at start + max_dist * dir.
        valid = (dir_z < -1e-6) & (starts_w[..., 2] > ground_z)
        t = torch.where(
            valid, (ground_z - starts_w[..., 2]) / dir_z, torch.full_like(dir_z, max_dist)
        )
        t = t.clamp(min=0.0, max=max_dist)
        hit_pos_w = starts_w + t.unsqueeze(-1) * dirs_w
        # When invalid, place hit_pos at the end of the unit-length ray.
        miss_pos = starts_w + max_dist * dirs_w
        hit_pos_w = torch.where(valid.unsqueeze(-1), hit_pos_w, miss_pos)
        normals_w = torch.zeros_like(hit_pos_w)
        normals_w[..., 2] = 1.0
        return t, hit_pos_w, normals_w

    def _scene_terrain(self) -> "TerrainImporter | None":
        """Return the active :class:`TerrainImporter`, or ``None`` if the env runs on flat ground.

        Fake / minimal envs (the unit-test suite) may not expose ``.scene``; treat that
        as "flat plane" rather than failing.
        """
        if self._env is None:
            return None
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return None
        return getattr(scene, "terrain", None)

    def _sample_heightfield(
        self,
        xy_world: torch.Tensor,
        terrain: "TerrainImporter",
    ) -> torch.Tensor:
        """Bilinearly sample terrain elevation at world ``(x, y)`` coordinates.

        ``xy_world`` is shape ``(..., 2)``; returns shape ``(...,)`` of world-frame z.
        Out-of-bounds samples are clamped to the height-field border.
        """
        device = str(xy_world.device)
        hf = terrain.heightfield_tensor(device, xy_world.dtype)  # (H, W) integer steps
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
        pos_x, pos_y, pos_z = terrain.terrain_origin

        # World -> heightfield grid coords.
        gx = (xy_world[..., 0] - pos_x) / h_scale
        gy = (xy_world[..., 1] - pos_y) / h_scale
        n_rows, n_cols = hf.shape
        gx = gx.clamp(0.0, float(n_rows - 1))
        gy = gy.clamp(0.0, float(n_cols - 1))

        gx0 = gx.floor().long()
        gy0 = gy.floor().long()
        gx1 = (gx0 + 1).clamp(max=n_rows - 1)
        gy1 = (gy0 + 1).clamp(max=n_cols - 1)
        fx = gx - gx0.to(gx.dtype)
        fy = gy - gy0.to(gy.dtype)

        v00 = hf[gx0, gy0]
        v10 = hf[gx1, gy0]
        v01 = hf[gx0, gy1]
        v11 = hf[gx1, gy1]
        v_x0 = v00 * (1.0 - fx) + v10 * fx
        v_x1 = v01 * (1.0 - fx) + v11 * fx
        v_xy = v_x0 * (1.0 - fy) + v_x1 * fy
        return v_xy * v_scale + pos_z
