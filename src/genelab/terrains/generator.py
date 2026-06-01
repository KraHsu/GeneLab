"""Terrain generator: dataclass cfg → 2D layout + Genesis kwargs.

The generator is a pure-Python helper: it builds the 2D ``subterrain_types`` layout
(random by proportion or explicit), aggregates the per-type parameters into the dict
shape Genesis expects, and computes the planar centers of each cell so callers can
place envs on them later (curriculum reset, locomotion goals).

Genesis keys ``subterrain_parameters`` by the *type string*, not per cell, so two
``SubTerrainCfg`` instances sharing a ``genesis_type`` at different parameters (e.g. a
RandomRough level curriculum ``rough_l0..l4``) would otherwise collapse to a single
parameter set — every cell built at the last-seen difficulty. :meth:`TerrainGenerator.
build_height_field` sidesteps that by generating each cell from its own parameters and
handing the finished array to ``Terrain(height_field=...)``; the importer always passes
it, so per-cell parameters are honoured regardless of type sharing.
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
    # When True (and ``layout`` is None), rows are ordered easiest → hardest by each
    # sub-terrain's ``difficulty`` (row 0 = easiest) instead of proportion-weighted random
    # tiling — so the env-level terrain curriculum (``mdp.terrain_levels_vel``) ramps real
    # difficulty as it promotes envs up the rows.
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
        names = list(cfg.sub_terrains.keys())
        if cfg.curriculum:
            # Difficulty-ordered rows (M3.3): sort the sub-terrains by ``difficulty`` and map
            # row 0 → easiest, last row → hardest, spreading evenly when the counts differ.
            # ``terrain_levels_vel`` indexes rows by the env's terrain level, so this makes
            # "promote to a higher level" mean "harder terrain". Every cell in a row uses the
            # row's single difficulty-ranked type (Genesis keys params by type, so a row can't
            # mix difficulties of the same type anyway). ``sorted`` is stable, so equal
            # difficulties keep their insertion order.
            ordered = sorted(names, key=lambda n: cfg.sub_terrains[n].difficulty)
            denom = max(cfg.num_rows - 1, 1)
            return [
                [ordered[round(i / denom * (len(ordered) - 1))]] * cfg.num_cols
                for i in range(cfg.num_rows)
            ]
        # Default: random tiling weighted by proportion.
        weights = [cfg.sub_terrains[n].proportion for n in names]
        if sum(weights) <= 0:
            raise ValueError("sub_terrains proportions sum to zero")
        rng = random.Random(cfg.seed)
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

    # ------------------------------------------------------------------ heightfield

    def build_height_field(self) -> Any:
        """Assemble the per-cell heightfield (raw integer units) for Genesis.

        Genesis keys ``subterrain_parameters`` by type string, so a layout that reuses
        one ``genesis_type`` at different difficulties (e.g. a RandomRough level
        curriculum ``rough_l0..l4``) collapses to a single param set -- every cell ends
        up at the last-seen difficulty. We sidestep that by generating each cell here
        with its own parameters (reusing Genesis' own isaacgym terrain generators) and
        handing the finished array to ``Terrain(height_field=...)``, which bypasses the
        type-keyed path entirely.

        Returns a ``(num_rows*(cell-1)+1, num_cols*(cell-1)+1)`` float array of heights
        in raw units (world z = value * ``vertical_scale``).
        """
        import numpy as np
        from genesis.ext.isaacgym import terrain_utils as _tu  # type: ignore[import-not-found]

        cfg = self.cfg
        hs, vs = cfg.horizontal_scale, cfg.vertical_scale
        per_r = int(cfg.subterrain_size[0] / hs + 1e-6) + 1
        per_c = int(cfg.subterrain_size[1] / hs + 1e-6) + 1
        rows = cfg.num_rows * (per_r - 1) + 1
        cols = cfg.num_cols * (per_c - 1) + 1
        hf = np.full((rows, cols), -np.inf, dtype=np.float32)

        saved = np.random.get_state()
        try:
            for i in range(cfg.num_rows):
                for j in range(cfg.num_cols):
                    sub_cfg = cfg.sub_terrains[self._layout[i][j]]
                    gtype = sub_cfg.genesis_type()
                    st = _tu.SubTerrain(
                        width=per_r, length=per_c, vertical_scale=vs, horizontal_scale=hs
                    )
                    # Deterministic, per-cell seed: reproducible across runs and varied
                    # across columns (Genesis' own path reseeds to 0 per cell, tiling an
                    # identical pattern into every cell of a row).
                    np.random.seed(cfg.seed + i * cfg.num_cols + j)
                    if gtype != "flat_terrain":
                        gen = getattr(_tu, gtype, None)
                        if gen is None:
                            raise ValueError(
                                f"build_height_field: no isaacgym generator for "
                                f"genesis_type {gtype!r} (cell {i},{j})"
                            )
                        gen(st, **sub_cfg.to_genesis_params())
                    data = st.height_field_raw
                    block = hf[
                        i * (per_r - 1) : (i + 1) * (per_r - 1) + 1,
                        j * (per_c - 1) : (j + 1) * (per_c - 1) + 1,
                    ]
                    block[:] = np.maximum(block, data)
        finally:
            np.random.set_state(saved)
        return hf
