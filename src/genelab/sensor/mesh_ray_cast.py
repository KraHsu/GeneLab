"""Genesis-backed mesh ray-cast sensor.

Wraps Genesis's native ``Raycaster`` (BVH ray-cast against scene meshes) as a GeneLab
:class:`~genelab.sensor.Sensor`, so rays intersect **arbitrary rigid bodies** — props,
obstacles, other robots — not just the ground.

This is the mesh counterpart to :class:`~genelab.sensor.RayCastSensor`. They coexist with a
clear division of responsibility:

* :class:`~genelab.sensor.RayCastSensor` — closed-form ray-vs-plane / ray-vs-heightfield,
  O(1) per ray. The fast path for terrain height-scan under a legged robot.
* :class:`MeshRayCastSensor` — Genesis BVH ray-cast against real triangle geometry. Use it
  when rays must hit actual meshes.

Only meshes whose material sets ``use_visual_raycasting=True`` are hit — see
:class:`~genelab.entity.RigidObjectCfg`.

GeneLab does not re-implement the ray math: the patterns and intersection are Genesis's. The
thin :class:`MeshGridPattern` / :class:`MeshSphericalPattern` specs only mirror Genesis's
pattern params and defer construction to sensor-build time, because Genesis patterns allocate
ray tensors in ``__init__`` (needing ``gs.init()``), which has not run when env cfgs are built.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class MeshGridPattern:
    """Deferred spec for Genesis's grid ray pattern (a regular XY grid of parallel rays).

    Mirrors ``genesis.options.sensors.GridPattern`` 1:1; :meth:`to_genesis` builds the real
    pattern at sensor-build time (after ``gs.init()``).
    """

    resolution: float = 0.1
    size: tuple[float, float] = (2.0, 2.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def to_genesis(self) -> Any:
        from genesis.options.sensors import GridPattern

        return GridPattern(resolution=self.resolution, size=self.size, direction=self.direction)


@dataclass
class MeshSphericalPattern:
    """Deferred spec for Genesis's spherical ray pattern (rays fanned over a FOV).

    Mirrors ``genesis.options.sensors.SphericalPattern``: give ``(n_points, fov)`` for uniform
    spacing by count, or ``angular_resolution`` to space by degrees. :meth:`to_genesis` builds
    the real pattern at sensor-build time.
    """

    fov: tuple[float | tuple[float, float], float | tuple[float, float]] = (360.0, 60.0)
    n_points: tuple[int, int] = (128, 64)
    angular_resolution: tuple[float | None, float | None] = (None, None)

    def to_genesis(self) -> Any:
        from genesis.options.sensors import SphericalPattern

        return SphericalPattern(
            fov=self.fov, n_points=self.n_points, angular_resolution=self.angular_resolution
        )


# A pattern spec is anything exposing ``to_genesis() -> <genesis RaycastPattern>``.
MeshRayPattern = MeshGridPattern | MeshSphericalPattern


@dataclass
class MeshRayCastSensorCfg(SensorCfg):
    """Configuration for :class:`MeshRayCastSensor`.

    The sensor attaches to ``link_name`` on ``entity_name`` and casts ``pattern``'s rays
    against the BVH of every mesh that opted in via ``use_visual_raycasting=True``. Distances
    clamp to ``max_range`` on a miss.
    """

    link_name: str = ""
    pattern: MeshRayPattern | None = None
    # Sensor pose relative to ``link_name``: ``pos_offset`` in the link's local frame,
    # ``euler_offset`` in degrees. Offset the rays clear of the robot's own body if you
    # don't want self-hits (the BVH includes the robot's collision faces).
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_range: float = 0.0
    max_range: float = 20.0
    # Genesis's built-in point-cloud debug draw: spheres at each ray's origin and hit point,
    # updated live in the viewer. ``draw_debug=True`` enables it; the colors / radius below
    # are forwarded to the Genesis raycaster.
    draw_debug: bool = False
    debug_sphere_radius: float = 0.02
    debug_ray_start_color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0)

    def build(self) -> "MeshRayCastSensor":
        return MeshRayCastSensor(self)


@dataclass
class MeshRayCastData:
    """Cached per-step output of :class:`MeshRayCastSensor`.

    Ray dimensions are flattened to a single ``num_rays`` axis regardless of the pattern
    (a grid's ``(nx, ny)`` and a sphere's ``(nh, nv)`` both collapse to ``num_rays``).
    """

    hit_pos_w: torch.Tensor  # (num_envs, num_rays, 3) world-frame hit points
    distances: torch.Tensor  # (num_envs, num_rays) ray travel distance, max_range on a miss


class MeshRayCastSensor(Sensor[MeshRayCastData]):
    """GeneLab wrapper over a Genesis ``Raycaster`` sensor (BVH mesh ray-cast)."""

    def __init__(self, cfg: MeshRayCastSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Register the Genesis raycaster before ``gs_scene.build`` (``add_sensor`` is
        ``@assert_unbuilt``). Mirrors the ``CameraSensor`` pre-build registration."""

        if not self._cfg_typed.link_name:
            raise ValueError(f"MeshRayCastSensorCfg(name={self._cfg.name!r}) requires link_name")
        if self._cfg_typed.pattern is None:
            raise ValueError(f"MeshRayCastSensorCfg(name={self._cfg.name!r}) requires a pattern")
        name = self._cfg_typed.entity_name
        wrapper = entities.get(name)
        if wrapper is None:
            raise ValueError(
                f"sensor {self._cfg.name!r}: pre_build_genesis needs entities[{name!r}]; "
                f"got {sorted(entities)!r}"
            )
        handle = getattr(wrapper, "gs_handle", None)
        if handle is None:
            raise RuntimeError(
                f"sensor {self._cfg.name!r}: entities[{name!r}].gs_handle is None; "
                "pre_build_genesis must run after the entity is spawned"
            )
        link_idx_local = self._resolve_link_idx(handle, self._cfg_typed.link_name)

        from genesis.options.sensors import Raycaster

        self._gs_sensor = gs_scene.add_sensor(
            Raycaster(
                pattern=self._cfg_typed.pattern.to_genesis(),
                entity_idx=handle.idx,
                link_idx_local=link_idx_local,
                pos_offset=self._cfg_typed.pos_offset,
                euler_offset=self._cfg_typed.euler_offset,
                min_range=self._cfg_typed.min_range,
                max_range=self._cfg_typed.max_range,
                return_world_frame=True,
                draw_debug=self._cfg_typed.draw_debug,
                debug_sphere_radius=self._cfg_typed.debug_sphere_radius,
                debug_ray_start_color=self._cfg_typed.debug_ray_start_color,
                debug_ray_hit_color=self._cfg_typed.debug_ray_hit_color,
            )
        )

    @staticmethod
    def _resolve_link_idx(handle: Any, link_name: str) -> int:
        names = [link.name for link in handle.links]
        try:
            return names.index(link_name)
        except ValueError as exc:
            raise ValueError(
                f"link {link_name!r} not found on entity; links={names!r}"
            ) from exc

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        if self._gs_sensor is None:
            raise RuntimeError(
                f"sensor {self._cfg.name!r}: Genesis raycaster not allocated. "
                "pre_build_genesis must run before bind (it registers the sensor pre-build)."
            )

    @property
    def gs_sensor(self) -> Any:
        """The underlying Genesis ``Raycaster`` handle (e.g. for built-in debug drawing)."""
        return self._gs_sensor

    @property
    def ray_starts_w(self) -> Any:
        """World-frame ray origins ``(num_rays, 3)`` for the current pose, or ``None`` before
        build. Useful for drawing ray segments (start → ``hit_pos_w``) in the viewer."""
        return None if self._gs_sensor is None else self._gs_sensor.ray_starts

    def _compute_data(self) -> MeshRayCastData:
        if self._gs_sensor is None or self._env is None:
            raise RuntimeError(f"sensor {self._cfg.name!r}: _compute_data called before bind")
        data = self._gs_sensor.read()  # genesis RaycasterData(points, distances)
        # Flatten the pattern's ray-grid dims (e.g. a grid's (nx, ny)) to a single ray axis
        # so the GeneLab payload is (num_envs, num_rays, ...) for every pattern.
        batch = data.points.shape[0]
        return MeshRayCastData(
            hit_pos_w=data.points.reshape(batch, -1, 3),
            distances=data.distances.reshape(batch, -1),
        )
