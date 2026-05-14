"""Terrain generator: dataclass cfg → 2D layout + Genesis kwargs.

The generator is a pure-Python helper: it builds the 2D ``subterrain_types`` layout
(random by proportion or explicit), aggregates the per-type parameters into the dict
shape Genesis expects, and computes the planar centers of each cell so callers can
place envs on them later (curriculum reset, locomotion goals).

Note: Genesis keys ``subterrain_parameters`` by the *type string*, not per cell. If
two ``SubTerrainCfg`` instances map to the same ``genesis_type`` (e.g. two
``PyramidStairsCfg`` with different ``step_width``), only one parameter set survives.
Use distinct keys in ``TerrainGeneratorCfg.sub_terrains`` for distinct geometries; for
in-type variation use ``randomize=True`` on the importer.
"""

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.terrains.sub_terrain import SubTerrainCfg


@dataclass
class TerrainGeneratorCfg:
    """Composition of a heightfield terrain grid.

    The grid is ``num_rows`` × ``num_cols`` cells; each cell has size
    ``subterrain_size`` (in meters). When ``layout`` is ``None``, cell types are
    sampled by ``SubTerrainCfg.proportion`` using a deterministic ``seed``.
    """

    num_rows: int = 4
    num_cols: int = 4
    subterrain_size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
    layout: tuple[tuple[str, ...], ...] | None = None
    curriculum: bool = False
    seed: int = 0


class TerrainGenerator:
    """Translate :class:`TerrainGeneratorCfg` into Genesis ``Terrain`` kwargs."""

    def __init__(self, cfg: TerrainGeneratorCfg) -> None:
        if not cfg.sub_terrains:
            raise ValueError("TerrainGeneratorCfg.sub_terrains must contain at least one entry")
        self.cfg = cfg
        self._layout: list[list[str]] = self._build_layout()
        self._env_origins: torch.Tensor = self._compute_origins()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> list[list[str]]:
        cfg = self.cfg
        if cfg.layout is not None:
            rows = [list(row) for row in cfg.layout]
            if len(rows) != cfg.num_rows or any(len(r) != cfg.num_cols for r in rows):
                raise ValueError(
                    f"layout shape {len(rows)}x{len(rows[0]) if rows else 0} "
                    f"does not match num_rows x num_cols = {cfg.num_rows}x{cfg.num_cols}"
                )
            unknown = {name for row in rows for name in row} - set(cfg.sub_terrains.keys())
            if unknown:
                raise ValueError(f"layout references unknown sub_terrains: {sorted(unknown)}")
            return rows
        # Random tiling weighted by proportion.
        names = list(cfg.sub_terrains.keys())
        weights = [cfg.sub_terrains[n].proportion for n in names]
        if sum(weights) <= 0:
            raise ValueError("sub_terrains proportions sum to zero")
        rng = random.Random(cfg.seed)
        # PR3 (curriculum) will refine row ordering by difficulty; for PR1 every row uses
        # the same proportion distribution regardless of ``cfg.curriculum``.
        return [rng.choices(names, weights=weights, k=cfg.num_cols) for _ in range(cfg.num_rows)]

    # ------------------------------------------------------------------ origins

    def _compute_origins(self) -> torch.Tensor:
        cfg = self.cfg
        sx, sy = cfg.subterrain_size
        px, py, pz = cfg.pos
        origins = torch.zeros((cfg.num_rows, cfg.num_cols, 3))
        for i in range(cfg.num_rows):
            for j in range(cfg.num_cols):
                origins[i, j, 0] = px + (i + 0.5) * sx
                origins[i, j, 1] = py + (j + 0.5) * sy
                origins[i, j, 2] = pz
        return origins

    # ------------------------------------------------------------------ outputs

    @property
    def layout(self) -> list[list[str]]:
        """2D list of sub-terrain cfg keys at each (row, col)."""
        return self._layout

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-cell planar centers, shape ``(num_rows, num_cols, 3)``.

        ``z`` is set to ``cfg.pos[2]`` (the terrain root height); callers needing the
        actual surface elevation should sample :attr:`TerrainImporter.heightfield`
        after spawn.
        """
        return self._env_origins

    @property
    def genesis_kwargs(self) -> dict[str, Any]:
        """Kwargs ready to splat into ``gs.morphs.Terrain(...)``."""
        cfg = self.cfg
        types_2d = [[cfg.sub_terrains[name].genesis_type() for name in row] for row in self._layout]
        params: dict[str, dict[str, Any]] = {}
        for sub_cfg in cfg.sub_terrains.values():
            gtype = sub_cfg.genesis_type()
            p = sub_cfg.to_genesis_params()
            if p:
                params[gtype] = p
        kwargs: dict[str, Any] = {
            "n_subterrains": (cfg.num_rows, cfg.num_cols),
            "subterrain_size": cfg.subterrain_size,
            "horizontal_scale": cfg.horizontal_scale,
            "vertical_scale": cfg.vertical_scale,
            "subterrain_types": types_2d,
            "pos": cfg.pos,
        }
        if params:
            kwargs["subterrain_parameters"] = params
        return kwargs
