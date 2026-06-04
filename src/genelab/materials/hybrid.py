"""Hybrid material wrapper (``gs.materials.Hybrid``): soft skin over a rigid skeleton."""

from dataclasses import dataclass
from typing import Any

from genelab.materials.base import MaterialCfg


@dataclass
class HybridMaterialCfg(MaterialCfg):
    """``gs.materials.Hybrid`` — a rigid skeleton coupled to a soft skin.

    ``material_rigid`` and ``material_soft`` are themselves material cfgs and are
    built recursively. :meth:`required_solvers` reports the union of both parts so the
    scene enables every solver the composite needs.
    """

    material_rigid: MaterialCfg | None = None
    material_soft: MaterialCfg | None = None
    use_default_coupling: bool | None = None
    damping: float | None = None
    thickness: float | None = None
    soft_dv_coef: float | None = None

    def required_solvers(self) -> set[str]:
        out: set[str] = set()
        if self.material_rigid is not None:
            out |= self.material_rigid.required_solvers()
        if self.material_soft is not None:
            out |= self.material_soft.required_solvers()
        return out

    def build(self, gs: Any) -> Any:
        if self.material_rigid is None or self.material_soft is None:
            raise ValueError(
                "HybridMaterialCfg requires both material_rigid and material_soft to be set"
            )
        scalar = {
            "use_default_coupling": self.use_default_coupling,
            "damping": self.damping,
            "thickness": self.thickness,
            "soft_dv_coef": self.soft_dv_coef,
        }
        return gs.materials.Hybrid(
            material_rigid=self.material_rigid.build(gs),
            material_soft=self.material_soft.build(gs),
            **{k: v for k, v in scalar.items() if v is not None},
        )
