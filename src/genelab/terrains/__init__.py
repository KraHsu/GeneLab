"""Terrain generation and terrain asset extension points."""

from genelab.terrains.deformable import (
    DeformableTerrain,
    DeformableTerrainCfg,
    DeformableTerrainDriver,
    NormalLaw,
    TerrainState,
    compliant_normal_force,
)
from genelab.terrains.footprint_map import FootprintMap
from genelab.terrains.generator import TerrainGenerator, TerrainGeneratorCfg
from genelab.terrains.importer import TerrainImporter
from genelab.terrains.sub_terrain import (
    DiscreteObstaclesCfg,
    FlatPatchCfg,
    FractalCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SlopeCfg,
    SteppingStonesCfg,
    SubTerrainCfg,
    WaveCfg,
)

__all__ = [
    "DeformableTerrain",
    "DeformableTerrainCfg",
    "DeformableTerrainDriver",
    "DiscreteObstaclesCfg",
    "FlatPatchCfg",
    "FootprintMap",
    "FractalCfg",
    "NormalLaw",
    "PyramidStairsCfg",
    "RandomRoughCfg",
    "SlopeCfg",
    "SteppingStonesCfg",
    "SubTerrainCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainImporter",
    "TerrainState",
    "WaveCfg",
    "compliant_normal_force",
]
