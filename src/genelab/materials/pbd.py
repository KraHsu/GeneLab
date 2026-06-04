"""Position-Based Dynamics material wrappers (``gs.materials.PBD.*``).

PBD bodies need ``pbd_options`` on the scene; it is enabled automatically when any
entity carries a PBD material. Tune it via
:class:`~genelab.materials.options.PbdOptionsCfg`.
"""

from dataclasses import dataclass
from typing import ClassVar

from genelab.materials.base import MaterialCfg


@dataclass
class _PbdMaterialCfg(MaterialCfg):
    gs_solver: ClassVar[str] = "pbd"
    rho: float | None = None


@dataclass
class PbdElasticCfg(_PbdMaterialCfg):
    """``gs.materials.PBD.Elastic`` — elastic deformable solid."""

    _gs_path: ClassVar[str] = "PBD.Elastic"
    static_friction: float | None = None
    kinetic_friction: float | None = None
    stretch_compliance: float | None = None
    bending_compliance: float | None = None
    volume_compliance: float | None = None
    stretch_relaxation: float | None = None
    bending_relaxation: float | None = None
    volume_relaxation: float | None = None


@dataclass
class PbdClothCfg(_PbdMaterialCfg):
    """``gs.materials.PBD.Cloth`` — PBD cloth (``rho`` is areal, kg/m²)."""

    _gs_path: ClassVar[str] = "PBD.Cloth"
    static_friction: float | None = None
    kinetic_friction: float | None = None
    stretch_compliance: float | None = None
    bending_compliance: float | None = None
    stretch_relaxation: float | None = None
    bending_relaxation: float | None = None
    air_resistance: float | None = None


@dataclass
class PbdLiquidCfg(_PbdMaterialCfg):
    """``gs.materials.PBD.Liquid`` — PBD fluid."""

    _gs_path: ClassVar[str] = "PBD.Liquid"
    sampler: str | None = None
    density_relaxation: float | None = None
    viscosity_relaxation: float | None = None


@dataclass
class PbdParticleCfg(_PbdMaterialCfg):
    """``gs.materials.PBD.Particle`` — free PBD particles."""

    _gs_path: ClassVar[str] = "PBD.Particle"
    sampler: str | None = None
