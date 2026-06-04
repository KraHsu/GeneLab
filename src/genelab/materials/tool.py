"""Tool material wrapper (``gs.materials.Tool``) for external grippers / actuators."""

from dataclasses import dataclass
from typing import ClassVar

from genelab.materials.base import MaterialCfg


@dataclass
class ToolMaterialCfg(MaterialCfg):
    """``gs.materials.Tool`` — externally-driven collision tool."""

    gs_solver: ClassVar[str] = "tool"
    _gs_path: ClassVar[str] = "Tool"
    friction: float | None = None
    coup_softness: float | None = None
    collision: bool | None = None
    sdf_res: int | None = None
