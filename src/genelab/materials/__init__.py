"""Declarative wrappers for Genesis materials, surfaces, and textures.

Every ``*Cfg`` mirrors a ``gs.materials`` / ``gs.surfaces`` / ``gs.textures``
constructor field-for-field; ``None`` means "use the Genesis default". Attach a
material/surface to a scene object via :class:`~genelab.entity.RigidObjectCfg`'s
``material`` / ``surface`` fields. Non-rigid materials cause
:class:`~genelab.scene.InteractiveScene` to auto-enable the matching solver options;
tune those via :class:`~genelab.materials.options.SolverOptionsCfg`.
"""

from genelab.materials.base import MaterialCfg
from genelab.materials.fem import FemClothCfg, FemElasticCfg, FemMuscleCfg
from genelab.materials.hybrid import HybridMaterialCfg
from genelab.materials.mpm import (
    MpmElasticCfg,
    MpmElastoPlasticCfg,
    MpmLiquidCfg,
    MpmMuscleCfg,
    MpmSandCfg,
    MpmSnowCfg,
)
from genelab.materials.options import (
    FemOptionsCfg,
    MpmOptionsCfg,
    PbdOptionsCfg,
    SfOptionsCfg,
    SolverOptionsCfg,
    SphOptionsCfg,
    ToolOptionsCfg,
)
from genelab.materials.pbd import PbdClothCfg, PbdElasticCfg, PbdLiquidCfg, PbdParticleCfg
from genelab.materials.rigid import KinematicMaterialCfg, RigidMaterialCfg
from genelab.materials.sf import SfSmokeCfg
from genelab.materials.sph import SphLiquidCfg
from genelab.materials.surfaces import (
    BsdfCfg,
    EmissionCfg,
    GlassCfg,
    MetalCfg,
    PlasticCfg,
    SurfaceCfg,
)
from genelab.materials.textures import ColorTextureCfg, ImageTextureCfg, TextureCfg
from genelab.materials.tool import ToolMaterialCfg

__all__ = [
    "BsdfCfg",
    "ColorTextureCfg",
    "EmissionCfg",
    "FemClothCfg",
    "FemElasticCfg",
    "FemMuscleCfg",
    "FemOptionsCfg",
    "GlassCfg",
    "HybridMaterialCfg",
    "ImageTextureCfg",
    "KinematicMaterialCfg",
    "MaterialCfg",
    "MetalCfg",
    "MpmElasticCfg",
    "MpmElastoPlasticCfg",
    "MpmLiquidCfg",
    "MpmMuscleCfg",
    "MpmOptionsCfg",
    "MpmSandCfg",
    "MpmSnowCfg",
    "PbdClothCfg",
    "PbdElasticCfg",
    "PbdLiquidCfg",
    "PbdOptionsCfg",
    "PbdParticleCfg",
    "PlasticCfg",
    "RigidMaterialCfg",
    "SfOptionsCfg",
    "SfSmokeCfg",
    "SolverOptionsCfg",
    "SphLiquidCfg",
    "SphOptionsCfg",
    "SurfaceCfg",
    "TextureCfg",
    "ToolMaterialCfg",
    "ToolOptionsCfg",
]
