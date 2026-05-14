# Terrains

`genelab.terrains` is a thin GeneLab-shaped wrapper over Genesis's native
`gs.morphs.Terrain` morph. Sub-terrain configs map onto Genesis's built-in subterrain
string types, `TerrainGenerator` assembles them into a 2D grid, and `TerrainImporter`
spawns the terrain into the scene plus tracks per-env curriculum state.

## Why a dedicated layer

Genesis ships the height-field terrain primitives, but its native API uses pydantic
options keyed by string names (`"pyramid_stairs_terrain"`, `"random_uniform_terrain"`).
A GeneLab cfg layer keeps the surface dataclass-shaped (matching the rest of the project),
gives every sub-terrain a typed Python class with explicit fields, and tracks per-env
spawn / level state so curriculum terms can promote and demote without touching the
Genesis handle directly.

## Five shipped sub-terrains

| Class | Genesis type | Parameters |
|---|---|---|
| `FlatPatchCfg` | `flat_terrain` | none |
| `PyramidStairsCfg` | `pyramid_stairs_terrain` | `step_width`, `step_height` |
| `RandomRoughCfg` | `random_uniform_terrain` | `min_height`, `max_height`, `step`, `downsampled_scale` |
| `SlopeCfg` | `sloped_terrain` | `slope` |
| `WaveCfg` | `wave_terrain` | `num_waves`, `amplitude` |

`PyramidStairsCfg(step_height=-0.1)` builds concentric descending steps; positive
`step_height` would invert the pyramid. `RandomRoughCfg` samples bumps at
`downsampled_scale` then upsamples, so the apparent feature size is independent of
`horizontal_scale`. `SlopeCfg(slope=-0.5)` tilts the patch uniformly (negative follows
Genesis's default downhill direction); `WaveCfg` lays sinusoidal undulations across
the patch — a useful intermediate difficulty between flat and pyramid-stairs. Four
more Genesis types (`pyramid_sloped`, `discrete_obstacles`, `stairs`, `stepping_stones`)
are not wrapped yet — extending `SubTerrainCfg` to expose them is a 20-line task.

## Composing a terrain grid

```python
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    TerrainGeneratorCfg,
)
from genelab.configs import InteractiveSceneCfg

scene = InteractiveSceneCfg(
    env_spacing=(0.0, 0.0),
    terrain=TerrainGeneratorCfg(
        num_rows=4,
        num_cols=4,
        subterrain_size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        sub_terrains={
            "flat": FlatPatchCfg(proportion=1.0),
            "stairs": PyramidStairsCfg(step_width=0.5, step_height=-0.08, proportion=2.0),
            "rough": RandomRoughCfg(min_height=-0.05, max_height=0.05, proportion=1.0),
        },
        curriculum=True,
        seed=0,
    ),
)
```

`sub_terrains` is a `dict[str, SubTerrainCfg]`; the key is the local cfg label, the
value carries the Genesis parameters. `layout` defaults to `None` (random tiling by
`proportion`); pass a `tuple[tuple[str, ...], ...]` of cfg keys to pin the grid
deterministically.

!!! warning "One parameter set per Genesis type"
    Genesis keys `subterrain_parameters` by the *type string*, not per cell. Two
    `PyramidStairsCfg` instances with different `step_width` collapse to a single
    parameter set. Use distinct keys for distinct geometries, or rely on
    `randomize=True` on the importer for in-type variation.

## Ray-cast sensor integration

When `InteractiveSceneCfg.terrain` is set, `InteractiveScene.build()` spawns
`gs.morphs.Terrain` instead of the default plane and exposes the importer as
`scene.terrain`. `RayCastSensor` checks for it and, when present, bilinearly samples
`terrain.heightfield_tensor` at each ray's world `(x, y)` to compute hit elevation.
`TerrainHeightSensor` is a thin wrapper around the same path, so the per-link height
scan output reflects the true terrain instead of a constant plane.

Vertical and near-vertical rays match the underlying height field to within
`vertical_scale`. Oblique rays use the elevation at the ray-start `(x, y)` — accurate
for height-scan grids but approximate for BVH-style queries; override
`RayCastSensor._intersect_world_rays` for those.

## Curriculum: `terrain_levels_vel`

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"distance_threshold": 2.0, "demote_ratio": 0.5},
    ),
}
```

Each reset, the term compares every env's walked distance (XY norm of `root_pos -
spawn_pos`) against `distance_threshold`. Envs that walked further than the threshold
move up one row of the difficulty grid (capped at `num_rows - 1`); envs that walked
less than `distance_threshold * demote_ratio` move down (floored at 0). The curriculum
then writes the new spawn pose into the sim via `Articulation.write_root_state`, so
the next episode starts on the new sub-terrain. The manager logs the mean level under
`Curriculum/terrain_levels`.

## Failure modes worth knowing

* **Empty `sub_terrains`** — `TerrainGenerator.__init__` raises immediately; the
  generator has nothing to tile.
* **`layout` shape mismatch** — `ValueError` lists the actual vs declared shape.
* **`layout` references unknown key** — `ValueError` lists the missing keys.
* **Per-env state access before build** — `terrain_levels` / `terrain_cols` /
  `spawn_pos` raise `RuntimeError` until `init_per_env_state` runs;
  `InteractiveScene.build` calls it automatically.
* **`heightfield` access before spawn** — `RuntimeError`; spawn first, then read.

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
- [Actuators](actuators.md)
