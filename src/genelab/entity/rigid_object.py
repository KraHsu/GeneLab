"""Rigid-object scene entity: Genesis static/dynamic body wrapper.

Mirrors :class:`Articulation` for non-articulated geometry (planes, primitives, meshes,
single-rigid MJCFs). Two-phase usage:

* ``spawn(gs_scene)`` creates the morph and adds it to the scene BEFORE ``scene.build()``.
* No ``bind`` step is needed — rigid objects don't carry per-step state in M1.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class RigidObjectCfg:
    """Description of a single rigid-body scene element.

    ``morph`` selects which Genesis primitive backs the object. ``file`` is required for
    ``"mesh"`` / ``"mjcf"`` morphs. ``size`` is interpreted per-morph: a 3-tuple for boxes,
    a 1-tuple radius for spheres.

    ``friction`` and ``density`` map onto ``gs.materials.Rigid(friction=..., rho=...)``;
    ``None`` keeps Genesis's defaults. Manipulation tasks usually need a non-default
    friction on the object being grasped — the Genesis fallback is too low to hold a
    cube between parallel fingers.

    ``use_visual_raycasting`` opts the object's visual mesh into Genesis's BVH ray-cast, so
    a :class:`~genelab.sensor.MeshRayCastSensor`'s rays can hit it. Off by default (the BVH
    only includes opted-in meshes).
    """

    morph: Literal["plane", "box", "sphere", "mesh", "mjcf"] = "plane"
    file: str | None = None
    size: tuple[float, ...] = ()
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fixed: bool = True
    friction: float | None = None
    density: float | None = None
    use_visual_raycasting: bool = False


class RigidObject:
    """Genesis static / rigid body wrapper."""

    def __init__(self, cfg: RigidObjectCfg, *, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self._gs_handle: Any = None

    def spawn(self, gs_scene: Any) -> None:
        """Pre-build: build the morph and add it to ``gs_scene``."""
        import genesis as gs  # type: ignore[import-not-found]

        morph_kind = self.cfg.morph
        if morph_kind == "plane":
            morph = gs.morphs.Plane(pos=tuple(self.cfg.init_pos))
        elif morph_kind == "box":
            if len(self.cfg.size) != 3:
                raise ValueError(
                    f"RigidObject(name={self.name!r}, morph='box') requires size=(x, y, z)"
                )
            morph = gs.morphs.Box(
                size=tuple(self.cfg.size),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
                fixed=self.cfg.fixed,
            )
        elif morph_kind == "sphere":
            if len(self.cfg.size) != 1:
                raise ValueError(
                    f"RigidObject(name={self.name!r}, morph='sphere') requires size=(radius,)"
                )
            morph = gs.morphs.Sphere(
                radius=float(self.cfg.size[0]),
                pos=tuple(self.cfg.init_pos),
                fixed=self.cfg.fixed,
            )
        elif morph_kind == "mesh":
            if not self.cfg.file:
                raise ValueError(f"RigidObject(name={self.name!r}, morph='mesh') requires file=...")
            morph = gs.morphs.Mesh(
                file=str(self.cfg.file),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
                fixed=self.cfg.fixed,
            )
        elif morph_kind == "mjcf":
            if not self.cfg.file:
                raise ValueError(f"RigidObject(name={self.name!r}, morph='mjcf') requires file=...")
            morph = gs.morphs.MJCF(
                file=str(self.cfg.file),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
            )
        else:  # pragma: no cover - exhaustive Literal
            raise ValueError(f"unknown rigid-object morph: {morph_kind!r}")
        material = self._material_or_default(gs)
        self._gs_handle = (
            gs_scene.add_entity(morph, material=material)
            if material is not None
            else gs_scene.add_entity(morph)
        )

    def _material_or_default(self, gs: Any) -> Any:
        """Build a ``gs.materials.Rigid`` only if the cfg overrides a default —
        otherwise return ``None`` so ``add_entity`` falls back to its built-in
        material and behavior stays unchanged for callers that don't opt in."""
        if (
            self.cfg.friction is None
            and self.cfg.density is None
            and not self.cfg.use_visual_raycasting
        ):
            return None
        kwargs: dict[str, Any] = {}
        if self.cfg.friction is not None:
            kwargs["friction"] = float(self.cfg.friction)
        if self.cfg.density is not None:
            kwargs["rho"] = float(self.cfg.density)
        if self.cfg.use_visual_raycasting:
            kwargs["use_visual_raycasting"] = True
        return gs.materials.Rigid(**kwargs)

    @property
    def gs_handle(self) -> Any:
        return self._gs_handle
