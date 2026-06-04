"""Surface (visual appearance) wrappers (``gs.surfaces.*``).

Surfaces control rendering only — they are orthogonal to the physics material. The
shortcut fields (``color``, ``roughness``, …) mirror Genesis's surface shortcuts;
``diffuse_texture`` accepts a :class:`~genelab.materials.textures.TextureCfg` that is
built recursively. A surface is attached to an entity via ``RigidObjectCfg.surface``.
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal

from genelab.materials.textures import TextureCfg


@dataclass
class SurfaceCfg:
    """Base surface — maps to ``gs.surfaces.Default`` (a BSDF) unless subclassed."""

    _gs_path: ClassVar[str] = "Default"
    color: tuple[float, ...] | None = None
    opacity: float | None = None
    roughness: float | None = None
    metallic: float | None = None
    emissive: tuple[float, ...] | None = None
    ior: float | None = None
    vis_mode: Literal["visual", "collision", "particle", "sdf", "recon"] | None = None
    diffuse_texture: TextureCfg | None = None

    def build(self, gs: Any) -> Any:
        ctor = getattr(gs.surfaces, type(self)._gs_path)
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            kwargs[f.name] = value.build(gs) if isinstance(value, TextureCfg) else value
        return ctor(**kwargs)


@dataclass
class GlassCfg(SurfaceCfg):
    """``gs.surfaces.Glass`` — transparent / refractive surface."""

    _gs_path: ClassVar[str] = "Glass"


@dataclass
class MetalCfg(SurfaceCfg):
    """``gs.surfaces.Metal`` — metallic surface with a named ``metal_type``."""

    _gs_path: ClassVar[str] = "Metal"
    metal_type: (
        Literal["aluminium", "gold", "copper", "brass", "iron", "titanium", "vanadium", "lithium"]
        | None
    ) = None


@dataclass
class PlasticCfg(SurfaceCfg):
    """``gs.surfaces.Plastic`` — matte / glossy dielectric surface."""

    _gs_path: ClassVar[str] = "Plastic"


@dataclass
class EmissionCfg(SurfaceCfg):
    """``gs.surfaces.Emission`` — emissive (glowing) surface."""

    _gs_path: ClassVar[str] = "Emission"


@dataclass
class BsdfCfg(SurfaceCfg):
    """``gs.surfaces.BSDF`` — general physically-based surface."""

    _gs_path: ClassVar[str] = "BSDF"
