"""Rigid and kinematic material wrappers (``gs.materials.Rigid`` / ``.Kinematic``)."""

from dataclasses import dataclass
from typing import ClassVar

from genelab.materials.base import MaterialCfg


@dataclass
class RigidMaterialCfg(MaterialCfg):
    """``gs.materials.Rigid`` — friction, density, and rigid-coupling parameters.

    ``friction`` is clamped by Genesis to ``[0.01, 5.0]``. ``rho`` is the density.
    The ``coup_*`` knobs tune two-way soft coupling with deformable solvers; leave
    them ``None`` for the Genesis defaults.
    """

    gs_solver: ClassVar[str] = "rigid"
    _gs_path: ClassVar[str] = "Rigid"
    friction: float | None = None
    rho: float | None = None
    use_visual_raycasting: bool | None = None
    needs_coup: bool | None = None
    coup_friction: float | None = None
    coup_softness: float | None = None
    coup_restitution: float | None = None
    gravity_compensation: float | None = None


@dataclass
class KinematicMaterialCfg(MaterialCfg):
    """``gs.materials.Kinematic`` — physics-inert, visualization-only geometry."""

    gs_solver: ClassVar[str] = "kinematic"
    _gs_path: ClassVar[str] = "Kinematic"
    use_visual_raycasting: bool | None = None
