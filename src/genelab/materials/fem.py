"""Finite Element Method material wrappers (``gs.materials.FEM.*``).

FEM bodies need ``fem_options`` on the scene; it is enabled automatically when any
entity carries an FEM material. Tune it via
:class:`~genelab.materials.options.FemOptionsCfg`.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from genelab.materials.base import MaterialCfg

type _FemModel = Literal["linear", "stable_neohookean", "linear_corotated"]


@dataclass
class _FemMaterialCfg(MaterialCfg):
    gs_solver: ClassVar[str] = "fem"
    E: float | None = None
    nu: float | None = None
    rho: float | None = None
    friction_mu: float | None = None


@dataclass
class FemElasticCfg(_FemMaterialCfg):
    """``gs.materials.FEM.Elastic`` — elastic deformable solid."""

    _gs_path: ClassVar[str] = "FEM.Elastic"
    model: _FemModel | None = None


@dataclass
class FemClothCfg(_FemMaterialCfg):
    """``gs.materials.FEM.Cloth`` — FEM cloth / fabric."""

    _gs_path: ClassVar[str] = "FEM.Cloth"
    thickness: float | None = None
    bending_stiffness: float | None = None
    model: _FemModel | None = None


@dataclass
class FemMuscleCfg(_FemMaterialCfg):
    """``gs.materials.FEM.Muscle`` — actuatable FEM muscle."""

    _gs_path: ClassVar[str] = "FEM.Muscle"
    model: _FemModel | None = None
    n_groups: int | None = None
