"""Stable-Fluid (smoke) material wrapper (``gs.materials.SF.*``).

Smoke needs ``sf_options`` on the scene; it is enabled automatically when any entity
carries an SF material. Tune it via :class:`~genelab.materials.options.SfOptionsCfg`.
"""

from dataclasses import dataclass
from typing import ClassVar

from genelab.materials.base import MaterialCfg


@dataclass
class SfSmokeCfg(MaterialCfg):
    """``gs.materials.SF.Smoke`` — Eulerian smoke / gas."""

    gs_solver: ClassVar[str] = "sf"
    _gs_path: ClassVar[str] = "SF.Smoke"
    sampler: str | None = None
