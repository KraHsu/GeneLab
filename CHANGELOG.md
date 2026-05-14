# Changelog

All notable changes to GeneLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is still on a 0.x
trajectory so breaking changes can land in any minor release until the 1.0 stabilisation.

## [Unreleased]

### Added

- `genelab.terrains` namespace with `SubTerrainCfg` (abstract) and five concrete
  sub-terrains: `FlatPatchCfg`, `PyramidStairsCfg`, `RandomRoughCfg`, `SlopeCfg`,
  `WaveCfg`. `TerrainGeneratorCfg` composes them into a 2D grid (random by proportion
  or explicit `layout`); `TerrainGenerator` emits Genesis kwargs and per-cell
  `env_origins`; `TerrainImporter` spawns `gs.morphs.Terrain` and exposes the
  post-build `heightfield` for sensors.
- `InteractiveSceneCfg.terrain` now accepts a `TerrainGeneratorCfg`; `InteractiveScene`
  routes through `TerrainImporter` when set, falling back to the default flat plane
  otherwise. `InteractiveScene.terrain` exposes the active importer to consumers.
- `RayCastSensor` now samples `scene.terrain.heightfield_tensor` bilinearly when a
  terrain is attached; flat-plane behaviour is preserved when `scene.terrain is None`
  or when the sensor runs against a fake env without a `scene` attribute (unit-test
  surface). `TerrainHeightSensor` inherits the new path through its inner ray-cast.
- `TerrainImporter` gains `heightfield_tensor(device, dtype)` (device-keyed cache),
  `horizontal_scale`, `vertical_scale`, and `terrain_origin` accessors used by the
  ray-cast sampler.
- `TerrainImporter` tracks per-env curriculum state via `terrain_levels` /
  `terrain_cols` / `spawn_pos`; `init_per_env_state(num_envs, device)` is called once
  by `InteractiveScene.build()` post-Genesis-build. `update_env_origins(env_ids)`
  recomputes spawn origins after a level change.
- `genelab.mdp.curriculums.terrain_levels_vel`: curriculum term that promotes /
  demotes each env's terrain level by how far it walked from spawn (XY distance vs
  configurable `distance_threshold` and `demote_ratio`), then writes the new spawn
  pose into the sim via `Articulation.write_root_state`.
- `docs/concepts/terrains.{en,zh}.md`: bilingual concept page covering the generator,
  importer, sensor integration, curriculum, and known failure modes.
- `genelab.actuator` namespace with `ActuatorBase` and three concrete electromechanical
  models: `ImplicitPDActuator` (Genesis-internal PD), `IdealPDActuator` (Python-side PD
  via `control_dofs_force`), `DCMotorActuator` (`IdealPD` plus driving-direction torque
  saturation).
- `ArticulationCfg.actuators: dict[str, ActuatorBaseCfg]` groups joints by regex and
  carries `stiffness` / `damping` / `effort_limit` / `velocity_limit` / `armature` /
  `friction` / `action_scale` per group.
- `Articulation.write_joint_targets_partial` routes per-step joint targets to each
  actuator group's declared channel (position-target or force).
- `Articulation.action_scale_tensor` aggregates per-group action scales for inheritance
  by `JointPositionAction` when its `scale` is left `None` (the new default).
- `docs/concepts/actuators.{en,zh}.md` covers the actuator design and failure modes.

### Changed

- **Breaking**: `ArticulationCfg.joint_kp` / `joint_kv` / `action_scale=dict` are removed;
  configure equivalent groups via `actuators`. The bundled inverted-pendulum and Unitree
  G1 examples migrate accordingly — G1 now ships six `DCMotorActuatorCfg` groups
  (5020 / 7520_14 / 7520_22 / 4010 / waist / ankle).
- **Breaking**: `Articulation.joint_kp` / `joint_kv` / `action_scale` properties and the
  matching `ManagerBasedRlEnv` delegates are removed. Read actuator parameters through
  `articulation.actuators[name]` instead.
- **Breaking**: `Articulation.bind` rejects an empty `actuators` dict and any actuator
  configuration that leaves an actuated joint uncovered or covered by more than one
  group. Passive joints need an explicit zero-gain `ImplicitPDActuatorCfg`.
- `JointPositionActionCfg.scale` defaults to `None`, in which case per-joint scale is
  inherited from `Articulation.action_scale_tensor`. Passing a `float` or `dict` still
  overrides.

### Removed

- `examples/unitree/.../g1/constants.py` no longer exports `ActuatorGroup`, `_fan_out`,
  `G1_JOINT_KP`, `G1_JOINT_KV`, or `G1_ACTION_SCALE` — superseded by `G1_ACTUATORS_CFG`.

## [0.1.0] — 2026-05-14

- M1 milestone: `InteractiveScene` + `Articulation` / `RigidObject` entities extracted
  from `ManagerBasedRlEnv`; `SimulationCfg` / `InteractiveSceneCfg` split.
- Sensor abstraction (`Sensor[T]`) with contact, body velocity, ray-cast, and terrain
  height sensors.
- Mouse-interaction viewer plugin.
- rsl_rl-based PPO training/replay for the inverted-pendulum and Unitree G1 examples.
