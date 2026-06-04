"""Material Point Method material wrappers (``gs.materials.MPM.*``).

MPM bodies need ``mpm_options`` on the scene (a bounded simulation domain); the
scene enables it automatically when any entity carries an MPM material (see
:class:`~genelab.scene.InteractiveScene`). Tune the domain via
:class:`~genelab.materials.options.MpmOptionsCfg`.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from genelab.materials.base import MaterialCfg


@dataclass
class _MpmMaterialCfg(MaterialCfg):
    gs_solver: ClassVar[str] = "mpm"
    E: float | None = None
    nu: float | None = None
    rho: float | None = None
    sampler: str | None = None


@dataclass
class MpmElasticCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.Elastic`` — elastic deformable solid."""

    _gs_path: ClassVar[str] = "MPM.Elastic"
    model: Literal["corotation", "neohooken"] | None = None


@dataclass
class MpmElastoPlasticCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.ElastoPlastic`` — plastically-deforming solid."""

    _gs_path: ClassVar[str] = "MPM.ElastoPlastic"
    yield_lower: float | None = None
    yield_higher: float | None = None
    use_von_mises: bool | None = None
    von_mises_yield_stress: float | None = None


@dataclass
class MpmLiquidCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.Liquid`` — MPM fluid."""

    _gs_path: ClassVar[str] = "MPM.Liquid"
    viscous: bool | None = None


@dataclass
class MpmSandCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.Sand`` — granular material."""

    _gs_path: ClassVar[str] = "MPM.Sand"
    friction_angle: float | None = None


@dataclass
class MpmSnowCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.Snow`` — deformable snow / powder."""

    _gs_path: ClassVar[str] = "MPM.Snow"
    yield_lower: float | None = None
    yield_higher: float | None = None


@dataclass
class MpmMuscleCfg(_MpmMaterialCfg):
    """``gs.materials.MPM.Muscle`` — actuatable muscle tissue."""

    _gs_path: ClassVar[str] = "MPM.Muscle"
    model: Literal["corotation", "neohooken"] | None = None
    n_groups: int | None = None
