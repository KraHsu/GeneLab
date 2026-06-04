# Materials

`genelab.materials` wraps Genesis's material, surface, and texture constructors as
declarative config dataclasses. A material cfg attaches to a scene object through
`RigidObjectCfg.material`; a non-rigid one turns the object deformable or fluid and
makes `InteractiveScene` enable the matching physics solver.

## Design

Each `*Cfg` mirrors a `gs.materials` / `gs.surfaces` / `gs.textures` constructor
field-for-field, using the Genesis parameter names verbatim (`friction`, `rho`, `E`,
`nu`, `sampler`, …). Every field defaults to `None`, meaning "use the Genesis
default": `build(gs)` forwards only the fields that were set, so an untouched cfg
produces an object identical to calling the Genesis constructor with no arguments.

The module imports no `genesis` at load time — `build(gs)` receives the `gs` module
as an argument — so `import genelab.configs` stays free of torch.

## Material catalog

| Family | Cfg classes | Solver |
|---|---|---|
| Rigid | `RigidMaterialCfg`, `KinematicMaterialCfg` | rigid |
| Tool | `ToolMaterialCfg` | tool |
| MPM | `MpmElasticCfg`, `MpmElastoPlasticCfg`, `MpmLiquidCfg`, `MpmSandCfg`, `MpmSnowCfg`, `MpmMuscleCfg` | mpm |
| FEM | `FemElasticCfg`, `FemClothCfg`, `FemMuscleCfg` | fem |
| PBD | `PbdElasticCfg`, `PbdClothCfg`, `PbdLiquidCfg`, `PbdParticleCfg` | pbd |
| SPH | `SphLiquidCfg` | sph |
| SF | `SfSmokeCfg` | sf |
| Composite | `HybridMaterialCfg` | union of its parts |

`HybridMaterialCfg` couples a rigid skeleton (`material_rigid`) to a soft skin
(`material_soft`); both are themselves material cfgs, built recursively.

## Attaching a material

```python
from genelab.configs import InteractiveSceneCfg
from genelab.entity import RigidObjectCfg
from genelab.materials import MpmElasticCfg, MetalCfg

scene = InteractiveSceneCfg(
    entities={
        "blob": RigidObjectCfg(
            morph="box",
            size=(0.1, 0.1, 0.1),
            material=MpmElasticCfg(E=3e5, nu=0.3),
            surface=MetalCfg(metal_type="gold"),
        )
    }
)
```

`material` takes precedence over the `friction` / `density` shortcut on
`RigidObjectCfg`; leaving `material` unset keeps that shortcut working exactly as
before. `surface` is orthogonal to physics — it controls rendering only
(`gs.surfaces.Glass` / `Metal` / `Plastic` / `Emission` / `BSDF`).

## Solver enablement

A deformable or fluid body is inert unless `gs.Scene` is built with the matching
solver options. `InteractiveScene` reads each entity material's `required_solvers()`
and enables exactly the solvers in use — with Genesis defaults — so a rigid-only
scene is unchanged.

Domain bounds and iteration counts are tuned through `SolverOptionsCfg` on
`InteractiveSceneCfg.solvers`:

```python
from genelab.materials import SolverOptionsCfg, MpmOptionsCfg

scene = InteractiveSceneCfg(
    entities={"blob": RigidObjectCfg(morph="box", material=MpmElasticCfg())},
    solvers=SolverOptionsCfg(mpm=MpmOptionsCfg(lower_bound=(-1, -1, 0), upper_bound=(1, 1, 1))),
)
```

A solver field set under `solvers` is also enabled even when no material requires it,
which is useful for emitters that add particles at runtime.

!!! warning "MPM / SPH / PBD domains are bounded"
    MPM, SPH, and PBD simulate inside a finite box. The Genesis defaults are small
    (MPM is roughly a unit cube); a body that leaves the domain is clamped. Set
    `lower_bound` / `upper_bound` on the matching options cfg to fit the scene.

## CLI overrides

Material and solver fields are ordinary dotted override paths, surfaced by
`genelab info` and settable on `play` / `train`:

```bash
genelab play TASK --env.scene.entities.blob.material.E 500000
genelab play TASK --env.scene.solvers.mpm.lower_bound -1,-1,0
```

## See also

- [Scene and entities](scene.md)
- [Configs](configs.md)
