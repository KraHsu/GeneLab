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
    """

    morph: Literal["plane", "box", "sphere", "mesh", "mjcf"] = "plane"
    file: str | None = None
    size: tuple[float, ...] = ()
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fixed: bool = True


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
        self._gs_handle = gs_scene.add_entity(morph)

    @property
    def gs_handle(self) -> Any:
        return self._gs_handle
