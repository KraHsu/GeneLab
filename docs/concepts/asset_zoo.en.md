# Asset Zoo

The asset zoo is a curated bundle of robot configs and downloadable assets shipped with GeneLab.
It is useful for examples and smoke tests, but downstream projects should still own their robot
configs in their own packages.

## Bundled assets

| Name | Factory |
|---|---|
| Unitree G1 | `UnitreeG1Cfg` |
| Unitree Go1 | `UnitreeGo1Cfg` |
| ANYmal C | `AnymalCCfg` |
| Franka Panda | `FrankaPandaCfg` |
| Cartpole | `CartpoleCfg` |
| G1 LAFAN1 motion | `g1_lafan1_dance1_subject2()` |

Importing `genelab.asset_zoo` registers bundled robots into `ROBOTS`. Asset downloads stay lazy and
only happen when a factory needs the actual file.

## Cache and verification

Downloaded assets live under the project cache. Asset helpers can verify checksums before returning
paths, so tasks can depend on stable local files after the first fetch.

## Where to continue

- [Module Map](../reference/module-map.md)
- [Unitree G1](../examples/unitree-g1.md)
- [API Reference](../api/reference.md)
