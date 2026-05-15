# Changelog

All notable changes to GeneLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is still on a 0.x
trajectory so breaking changes can land in any minor release until the 1.0 stabilisation.

## [Unreleased]

### Added

- `genelab.asset_zoo` now ships five built-in robots: `CartpoleCfg`, `FrankaPandaCfg`,
  `UnitreeG1Cfg` (29-DoF humanoid, 6 DCMotor groups mirroring
  `examples/unitree/.../g1/constants.py`), `UnitreeGo1Cfg` (12-DoF quadruped, 3
  ImplicitPD groups aligned to Isaac Lab Go1 defaults), and `AnymalCCfg` (12-DoF
  quadruped, single ImplicitPD group aligned to Isaac Lab Anymal C defaults). The three
  Menagerie-sourced robots ship as `.tar.gz` bundles in `KraHsu/genelab-assets`,
  preserving BSD-3-Clause attribution.
- `AssetSpec.archive_member`: optional field that switches `fetch_asset` from
  single-file to archive mode. The URL is treated as a `.tar.gz`, the blob's md5 is
  verified end-to-end, and `tarfile`'s `data` filter extracts the contents into
  `<md5>/extracted/` (rejects symlinks, absolute paths, and parent-directory escapes).
  The returned path is `<md5>/extracted/<archive_member>` — Menagerie mesh references
  resolve automatically because the full folder structure is preserved.
- `genelab.asset_zoo` namespace with the first two built-in robot configurations,
  `CartpoleCfg` and `FrankaPandaCfg`. Both are registered into `ROBOTS` as an import
  side-effect of `load_builtin_registries()`; factories stay lazy so `genelab list
  robots` never touches the network. Cartpole gain values mirror
  `examples/inverted_pendulum` (`stiffness=80`, passive pole); Franka follows Isaac
  Lab's high-PD configuration (`panda_arm` k=400, `panda_hand` k=1e4) with the
  Menagerie home pose.
- `genelab.utils.download.fetch_asset` + `AssetSpec` + `AssetDownloadError`: md5-verified
  download helper for asset zoo entries. Files cache under
  `<project_root>/.cache/assets/<name>/<md5>/<filename>` via atomic stage-and-rename;
  md5 mismatches and `URLError` failures raise `AssetDownloadError` with both expected
  and actual digests in the message.
- `docs/concepts/asset_zoo.{en,zh}.md`: bilingual concept page covering the asset
  lifecycle, cache layout, md5 verification, and the template for adding new robot
  configurations.
- `FrameTransformerSensor` (with `FrameTransformerSensorCfg`, `TargetFrameCfg`, and
  `FrameTransformerData`): stateless forward-kinematics probe that outputs the world-frame
  pose AND source-frame pose for one or more target frames relative to a single source
  frame. Both source and target support rigid local-frame offsets (position + quaternion).
  `target_frames` is order-preserving so observation terms can slice by position; the
  matching `target_names: tuple[str, ...]` attribute exposes column identities. `bind`
  rejects unknown link names and empty `target_frames` (mirrors the contact-sensor
  unresolved-link error path).
- `IMUSensor` (with `IMUSensorCfg` and `IMUData`) sits alongside `BodyVelocitySensor` and
  outputs orientation, body-frame projected unit gravity, and body-frame linear / angular
  acceleration at a site rigidly attached to a link. Accelerations are computed by finite
  difference of the world-frame velocity buffers; the first control step after each
  `reset` returns zero acceleration (deliberate — avoids a spurious spike from a stale
  prev buffer). `gravity_bias=True` (default) follows the real-IMU specific-force
  convention. Per-env `bias_range_lin_acc` / `bias_range_ang_acc` resample on every reset.
- `RingPattern` and `HemispherePattern` ray-cast patterns alongside the existing
  `GridPattern`. `RingPattern` covers multi-line LIDAR sweeps (`num_horizontal × num_vertical`
  rays, automatic wrap-around handling on a 360° span); `HemispherePattern` distributes rays
  on a spherical cap via a Fibonacci lattice with configurable pole axis and half-angle.
  `RayCastSensorCfg.pattern` is now typed `GridPattern | RingPattern | HemispherePattern`.
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
