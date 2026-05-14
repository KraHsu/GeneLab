"""Unit tests for :mod:`genelab.terrains`.

The generator is Genesis-free: it builds a 2D layout and kwargs dict for
``gs.morphs.Terrain``. The importer is gated on Genesis runtime availability via
the ``genesis_runtime`` fixture (skipped when ``libEGL`` is missing, e.g. on the
default headless CI runner).
"""

from collections import Counter
from typing import Any

import pytest
import torch

from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SlopeCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainImporter,
    WaveCfg,
)


def test_sub_terrain_cfgs_emit_genesis_strings() -> None:
    assert FlatPatchCfg().genesis_type() == "flat_terrain"
    assert PyramidStairsCfg().genesis_type() == "pyramid_stairs_terrain"
    assert RandomRoughCfg().genesis_type() == "random_uniform_terrain"
    assert SlopeCfg().genesis_type() == "sloped_terrain"
    assert WaveCfg().genesis_type() == "wave_terrain"


def test_sub_terrain_params_round_trip() -> None:
    p = PyramidStairsCfg(step_width=0.5, step_height=-0.2)
    assert p.to_genesis_params() == {"step_width": 0.5, "step_height": -0.2}
    r = RandomRoughCfg(min_height=-0.05, max_height=0.05, step=0.05, downsampled_scale=0.25)
    assert r.to_genesis_params() == {
        "min_height": -0.05,
        "max_height": 0.05,
        "step": 0.05,
        "downsampled_scale": 0.25,
    }
    s = SlopeCfg(slope=-0.3)
    assert s.to_genesis_params() == {"slope": -0.3}
    w = WaveCfg(num_waves=3.0, amplitude=0.2)
    assert w.to_genesis_params() == {"num_waves": 3.0, "amplitude": 0.2}


def test_generator_rejects_empty_sub_terrains() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        TerrainGenerator(TerrainGeneratorCfg())


def test_generator_random_layout_respects_proportion_and_seed() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=10,
        num_cols=10,
        sub_terrains={
            "flat": FlatPatchCfg(proportion=0.0),
            "stairs": PyramidStairsCfg(proportion=1.0),
        },
        seed=42,
    )
    gen = TerrainGenerator(cfg)
    flat_cells = sum(name == "flat" for row in gen.layout for name in row)
    stairs_cells = sum(name == "stairs" for row in gen.layout for name in row)
    assert flat_cells == 0
    assert stairs_cells == 100

    cfg2 = TerrainGeneratorCfg(
        num_rows=8,
        num_cols=8,
        sub_terrains={
            "flat": FlatPatchCfg(proportion=1.0),
            "stairs": PyramidStairsCfg(proportion=1.0),
            "rough": RandomRoughCfg(proportion=2.0),
        },
        seed=1,
    )
    gen2 = TerrainGenerator(cfg2)
    counts = Counter(name for row in gen2.layout for name in row)
    # rough has 2x the weight; assert it dominates without depending on exact RNG draw.
    assert counts["rough"] > counts["flat"]
    assert counts["rough"] > counts["stairs"]
    assert sum(counts.values()) == 64

    # Same seed reproduces the same layout.
    gen2b = TerrainGenerator(cfg2)
    assert gen2.layout == gen2b.layout


def test_generator_explicit_layout() -> None:
    layout = (
        ("flat", "stairs"),
        ("stairs", "rough"),
    )
    cfg = TerrainGeneratorCfg(
        num_rows=2,
        num_cols=2,
        sub_terrains={
            "flat": FlatPatchCfg(),
            "stairs": PyramidStairsCfg(),
            "rough": RandomRoughCfg(),
        },
        layout=layout,
    )
    gen = TerrainGenerator(cfg)
    assert gen.layout == [["flat", "stairs"], ["stairs", "rough"]]


def test_generator_layout_shape_mismatch_raises() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=2,
        num_cols=3,
        sub_terrains={"flat": FlatPatchCfg()},
        layout=(("flat", "flat"), ("flat", "flat")),
    )
    with pytest.raises(ValueError, match="does not match"):
        TerrainGenerator(cfg)


def test_generator_layout_unknown_name_raises() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=1,
        num_cols=1,
        sub_terrains={"flat": FlatPatchCfg()},
        layout=(("missing",),),
    )
    with pytest.raises(ValueError, match="unknown sub_terrains"):
        TerrainGenerator(cfg)


def test_env_origins_shape_and_center_positions() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=4,
        num_cols=4,
        subterrain_size=(6.0, 6.0),
        pos=(0.0, 0.0, 0.0),
        sub_terrains={"flat": FlatPatchCfg()},
    )
    gen = TerrainGenerator(cfg)
    origins = gen.env_origins
    assert origins.shape == (4, 4, 3)
    # Cell (0, 0) center at (0+0.5)*6 = 3.0; cell (3, 3) center at (3+0.5)*6 = 21.0.
    assert torch.allclose(origins[0, 0], torch.tensor([3.0, 3.0, 0.0]))
    assert torch.allclose(origins[3, 3], torch.tensor([21.0, 21.0, 0.0]))


def test_genesis_kwargs_shape() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=2,
        num_cols=3,
        subterrain_size=(8.0, 8.0),
        horizontal_scale=0.2,
        vertical_scale=0.01,
        sub_terrains={
            "flat": FlatPatchCfg(),
            "stairs": PyramidStairsCfg(step_width=0.5, step_height=-0.15),
        },
        layout=(
            ("flat", "stairs", "flat"),
            ("stairs", "flat", "stairs"),
        ),
    )
    kw = TerrainGenerator(cfg).genesis_kwargs
    assert kw["n_subterrains"] == (2, 3)
    assert kw["subterrain_size"] == (8.0, 8.0)
    assert kw["horizontal_scale"] == 0.2
    assert kw["vertical_scale"] == 0.01
    assert kw["subterrain_types"] == [
        ["flat_terrain", "pyramid_stairs_terrain", "flat_terrain"],
        ["pyramid_stairs_terrain", "flat_terrain", "pyramid_stairs_terrain"],
    ]
    assert kw["subterrain_parameters"] == {
        "pyramid_stairs_terrain": {"step_width": 0.5, "step_height": -0.15},
    }


def test_importer_heightfield_blocks_pre_spawn() -> None:
    cfg = TerrainGeneratorCfg(
        num_rows=2,
        num_cols=2,
        sub_terrains={"flat": FlatPatchCfg()},
    )
    importer = TerrainImporter(cfg)
    with pytest.raises(RuntimeError, match="before spawn"):
        _ = importer.heightfield
    with pytest.raises(RuntimeError, match="before spawn"):
        _ = importer.terrain_scale


def test_importer_spawns_and_exposes_height_field(genesis_runtime: Any) -> None:
    gs = genesis_runtime
    cfg = TerrainGeneratorCfg(
        num_rows=2,
        num_cols=2,
        subterrain_size=(6.0, 6.0),
        horizontal_scale=0.25,
        vertical_scale=0.005,
        sub_terrains={
            "flat": FlatPatchCfg(),
            "stairs": PyramidStairsCfg(step_width=0.75, step_height=-0.1),
        },
        layout=(("flat", "stairs"), ("stairs", "flat")),
    )
    importer = TerrainImporter(cfg)
    scene = gs.Scene(show_viewer=False)
    importer.spawn(scene)
    scene.build(n_envs=1)
    hf = importer.heightfield
    # 2 cells * 6m / 0.25 horizontal_scale = 48 cells per axis + 1 for boundary.
    assert hf.shape == (49, 49)
    ts = importer.terrain_scale
    assert pytest.approx(float(ts[0])) == 0.25
    assert pytest.approx(float(ts[1])) == 0.005
