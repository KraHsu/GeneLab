"""Smoothed Particle Hydrodynamics material wrapper (``gs.materials.SPH.*``).

SPH bodies need ``sph_options`` on the scene; it is enabled automatically when any
entity carries an SPH material. Tune it via
:class:`~genelab.materials.options.SphOptionsCfg`.
"""

from dataclasses import dataclass
from typing import ClassVar

from genelab.materials.base import MaterialCfg


@dataclass
class SphLiquidCfg(MaterialCfg):
    """``gs.materials.SPH.Liquid`` — weakly-compressible SPH fluid."""

    gs_solver: ClassVar[str] = "sph"
    _gs_path: ClassVar[str] = "SPH.Liquid"
    rho: float | None = None
    stiffness: float | None = None
    exponent: float | None = None
    mu: float | None = None
    gamma: float | None = None
    sampler: str | None = None
