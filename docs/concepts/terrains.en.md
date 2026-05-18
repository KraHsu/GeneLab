# Terrains

GeneLab terrains are generated height-field grids that can be imported into Genesis scenes and read
by ray-based sensors and curricula.

## Why a terrain layer

Locomotion tasks need terrain geometry, spawn origins, difficulty levels, and height scans to agree.
Keeping terrain generation in one layer lets the scene, curriculum, and sensors share the same
source of truth.

## Building blocks

| Config | Shape |
|---|---|
| `FlatPatchCfg` | Flat sub-terrain. |
| `PyramidStairsCfg` | Stair pattern. |
| `RandomRoughCfg` | Random rough surface. |
| `SlopeCfg` | Sloped patch. |
| `WaveCfg` | Wave-like height field. |
| `TerrainGeneratorCfg` | Grid layout and selected sub-terrains. |
| `TerrainImporter` | Runtime bridge into Genesis. |

## Integration points

| Consumer | Uses terrain for |
|---|---|
| `InteractiveScene` | Adds generated geometry to the Genesis scene. |
| `TerrainHeightSensor` | Samples heights under a ray grid. |
| `terrain_levels_vel` curriculum | Promotes/demotes env difficulty by performance. |

## Where to continue

- [Sensors](sensors.md)
- [Managers and MDP terms](managers.md)
- [API Reference](../api/reference.md)
