"""Ray-cast sensor with a configurable pattern, attached to a robot link.

Two intersection backends:

* **Flat plane** (default when ``scene.terrain is None``): every ray hits the horizontal
  plane at ``ground_height``. Closed-form intersection; rays going up or parallel are
  treated as misses and clamp to ``max_distance``.
* **Height field** (when ``scene.terrain`` is a :class:`TerrainImporter`): bilinearly
  samples ``terrain.heightfield_tensor`` at each ray's ``(x, y)`` origin and uses the
  sampled elevation as the hit plane height. Accurate for vertical / near-vertical rays
  (the dominant case for height-scan grids); subclasses can override
  ``_intersect_world_rays`` for non-trivial BVH ray-tracing.

Three built-in patterns share an informal protocol (``num_rays() -> int`` and
``generate(device) -> (starts, dirs)``):

* :class:`GridPattern` — 2D rectangular grid, all rays parallel. Default; mirrors the reference.
* :class:`RingPattern` — multi-line LIDAR-style sweep around the +z axis.
* :class:`HemispherePattern` — Fibonacci-distributed rays on a spherical cap.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_state
from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
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


def _evenly_spaced_angles_rad(
    start_deg: float, end_deg: float, n: int, device: str
) -> torch.Tensor:
    """Return ``n`` azimuths in radians spanning ``[start_deg, end_deg]``.

    A full ±360° span is treated as a wrap-around and the closing endpoint is dropped
    (``linspace(..., n+1)[:-1]``) so a 360° sweep with ``n`` rays does not duplicate the
    first/last angle. Any other span uses inclusive linspace.
    """
    span = end_deg - start_deg
    if n <= 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    if abs(abs(span) - 360.0) < 1e-6:
        angles_deg = torch.linspace(start_deg, end_deg, n + 1, device=device)[:-1]
    else:
        angles_deg = torch.linspace(start_deg, end_deg, n, device=device)
    return angles_deg * (math.pi / 180.0)


@dataclass
class RingPattern:
    """Multi-line LIDAR sweep: ``num_horizontal`` azimuths × ``num_vertical`` elevations.

    All rays emit from the sensor origin (``starts = zeros``). ``horizontal_fov_deg`` and
    ``vertical_fov_deg`` are inclusive on both endpoints except when the horizontal span is
    exactly ±360° — then the closing azimuth is dropped to avoid a duplicate ray. A single
    horizontal line (``num_vertical=1``) at ``vertical_fov_deg=(0, 0)`` gives a planar lidar.
    """

    num_horizontal: int = 32
    num_vertical: int = 1
    horizontal_fov_deg: tuple[float, float] = (-180.0, 180.0)
    vertical_fov_deg: tuple[float, float] = (-15.0, 15.0)

    def num_rays(self) -> int:
        return self.num_horizontal * self.num_vertical

    def generate(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        n_h = self.num_horizontal
        n_v = self.num_vertical
        azimuth = _evenly_spaced_angles_rad(
            self.horizontal_fov_deg[0], self.horizontal_fov_deg[1], n_h, device
        )
        if n_v == 1:
            elevation = torch.tensor(
                [(self.vertical_fov_deg[0] + self.vertical_fov_deg[1]) * 0.5 * math.pi / 180.0],
                device=device,
                dtype=torch.float32,
            )
        else:
            elevation = torch.linspace(
                self.vertical_fov_deg[0],
                self.vertical_fov_deg[1],
                n_v,
                device=device,
            ) * (math.pi / 180.0)
        # Outer product over (vertical, horizontal); reshape so vertical varies slowest.
        el_grid, az_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        cos_el = torch.cos(el_grid).reshape(-1)
        sin_el = torch.sin(el_grid).reshape(-1)
        cos_az = torch.cos(az_grid).reshape(-1)
        sin_az = torch.sin(az_grid).reshape(-1)
        dirs = torch.stack([cos_el * cos_az, cos_el * sin_az, sin_el], dim=-1).contiguous()
        starts = torch.zeros_like(dirs)
        return starts, dirs


def _fibonacci_cap_dirs_z_aligned(n: int, polar_fov_deg: float, device: str) -> torch.Tensor:
    """Return ``(n, 3)`` unit vectors evenly distributed on a spherical cap around +z.

    Uses a Fibonacci lattice with area-uniform polar spacing on the cap. ``polar_fov_deg``
    is the half-angle from the pole; 90° covers a full hemisphere, 180° a full sphere.
    """
    fov_rad = polar_fov_deg * math.pi / 180.0
    cos_theta_min = math.cos(fov_rad)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    i = torch.arange(n, device=device, dtype=torch.float32)
    cos_theta = 1.0 - (i + 0.5) / float(n) * (1.0 - cos_theta_min)
    sin_theta = (1.0 - cos_theta * cos_theta).clamp(min=0.0).sqrt()
    phi = i * golden_angle
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    return torch.stack([x, y, z], dim=-1)


def _rotate_z_to(z_aligned: torch.Tensor, target_axis: tuple[float, float, float]) -> torch.Tensor:
    """Rotate ``(N, 3)`` vectors expressed in a frame where +z is the pole onto a frame
    where the pole has been mapped to ``target_axis`` (unit vector after normalisation)."""
    device = z_aligned.device
    dtype = z_aligned.dtype
    target = torch.tensor(target_axis, device=device, dtype=dtype)
    target = target / target.norm().clamp_min(1e-9)
    cos_a = float(target[2].item())
    if cos_a > 1.0 - 1e-9:
        return z_aligned
    if cos_a < -1.0 + 1e-9:
        # 180° rotation about +x: (x, y, z) → (x, -y, -z).
        flipped = z_aligned.clone()
        flipped[..., 1] = -flipped[..., 1]
        flipped[..., 2] = -flipped[..., 2]
        return flipped
    # Rodrigues' rotation: axis = z × target, angle = arccos(target · z).
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    axis = torch.linalg.cross(z_unit, target)
    axis = axis / axis.norm().clamp_min(1e-9)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    cross = torch.linalg.cross(axis.expand_as(z_aligned), z_aligned)
    dot = (axis * z_aligned).sum(dim=-1, keepdim=True)
    return z_aligned * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)


@dataclass
class HemispherePattern:
    """Fibonacci-distributed rays on a spherical cap centred on ``pole_axis``.

    All rays emit from the sensor origin (``starts = zeros``). ``polar_fov_deg`` is the
    half-angle from the pole: 90° is a full hemisphere, 180° a full sphere. ``num_rays``
    is the exact ray count produced by ``generate()`` (no Fibonacci rounding involved).
    """

    num_rays_target: int = 64
    pole_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    polar_fov_deg: float = 90.0

    def num_rays(self) -> int:
        return self.num_rays_target

    def generate(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        dirs_z = _fibonacci_cap_dirs_z_aligned(self.num_rays_target, self.polar_fov_deg, device)
        dirs = _rotate_z_to(dirs_z, self.pole_axis).contiguous()
        starts = torch.zeros_like(dirs)
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

    ``link_name`` anchors the pattern. ``attach_yaw_only`` keeps the pattern axis-aligned to
    the horizon when the link rolls / pitches (typical for terrain height scans on a torso
    link); set ``False`` to follow the full link orientation. ``ground_height`` is the z of
    the flat plane the default backend intersects against; subclasses ignore it.

    ``pattern`` accepts any of the bundled pattern dataclasses (:class:`GridPattern`,
    :class:`RingPattern`, :class:`HemispherePattern`); custom patterns satisfying the same
    informal protocol (``num_rays() -> int`` and ``generate(device) -> (starts, dirs)``)
    are accepted but should be re-exported through ``genelab.sensor`` for typing parity.
    """

    link_name: str = ""
    pattern: GridPattern | RingPattern | HemispherePattern = field(default_factory=GridPattern)
    attach_yaw_only: bool = True
    max_distance: float = 10.0
    ground_height: float = 0.0
    link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local-frame offset of the ray-pattern origin relative to ``link_name``.

    reference parity: foot height scans anchor at the ``left_foot``/``right_foot``
    site (offset ``(0.04, 0, -0.037)`` from ``ankle_roll_link`` in G1's XML),
    not at the link origin. Set to non-zero when you need site-frame ray casts.
    """

    def build(self) -> "RayCastSensor":
        return RayCastSensor(self)


class RayCastSensor(Sensor[RayCastData]):
    def __init__(self, cfg: RayCastSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx: int = -1
        self._ray_starts_local: torch.Tensor | None = None
        self._ray_dirs_local: torch.Tensor | None = None
        self._link_offset_local: torch.Tensor | None = None

    @property
    def num_rays(self) -> int:
        return self._cfg_typed.pattern.num_rays()

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        if not self._cfg_typed.link_name:
            raise ValueError(f"RayCastSensorCfg(name={self._cfg.name!r}) requires link_name")
        link_names = entity_articulation(env, self._cfg.entity_name).link_names
        try:
            self._link_idx = link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={link_names!r}"
            ) from exc
        self._ray_starts_local, self._ray_dirs_local = self._cfg_typed.pattern.generate(env.device)
        self._link_offset_local = torch.tensor(
            self._cfg_typed.link_offset, dtype=torch.float32, device=env.device
        )

    def _compute_data(self) -> RayCastData:
        assert (
            self._env is not None
            and self._ray_starts_local is not None
            and self._ray_dirs_local is not None
            and self._link_offset_local is not None
        )
        rs = entity_state(self._env, self._cfg.entity_name)
        link_pos = rs.link_pos[:, self._link_idx]
        link_quat = rs.link_quat_w[:, self._link_idx]
        rot_q = yaw_quat(link_quat) if self._cfg_typed.attach_yaw_only else link_quat
        # Project the local pattern (shifted by ``link_offset`` so the ray fan anchors
        # at the configured site rather than the link origin) into world coordinates.
        b = link_pos.shape[0]
        m = self._ray_starts_local.shape[0]
        q_expanded = rot_q.unsqueeze(1).expand(b, m, 4).contiguous()
        starts_local = self._ray_starts_local + self._link_offset_local
        starts_local_b = starts_local.unsqueeze(0).expand(b, m, 3).contiguous()
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
        # Match the Genesis collision mesh exactly: ``convert_heightfield_to_trimesh``
        # splits every cell on the (i, j)-(i+1, j+1) diagonal into two triangles, so the
        # ground is piecewise-linear *per triangle*, not bilinear over the quad. Sampling
        # the triangle the ray falls in keeps the sensor's clearance reading consistent
        # with where the feet actually contact — bilinear under-reports a step edge by up
        # to ``step/4`` and makes feet sink / corrupts foot-height obs on rough terrain.
        lower = fx >= fy  # (i, j)->(i+1, j)->(i+1, j+1) triangle, vs the (i, j+1) one
        h_lower = v00 + fx * (v10 - v00) + fy * (v11 - v10)
        h_upper = v00 + fy * (v01 - v00) + fx * (v11 - v01)
        v_xy = torch.where(lower, h_lower, h_upper)
        return v_xy * v_scale + pos_z
