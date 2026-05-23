# Changelog

All notable changes to GeneLab are recorded here.

## [Unreleased]

_Nothing yet._

## [0.2.0] — 2026-05-23

Second development release.

### Added

- **Reproducibility:** added deterministic evaluation with `genelab eval`, model export with
  `genelab export`, multi-seed training with `genelab train --seeds a,b,c [--parallel N]`,
  periodic in-training evaluation with `--eval-every K`, best-model saving, RGB-D camera
  sensing, and five bundled asset-zoo robots.
- **Sim2real hardening:** added actuator and joint domain randomization, interval-mode
  domain randomization, out-of-limit terminations, reward hard-constraints, the
  `MlpResidualActuator`, and observation-noise models including `ScaledNoise`,
  `CorrelatedNoise`, and `BiasDrift`.
- **Platform breadth:** added new sub-terrain types, difficulty-ordered terrain curriculum,
  camera segmentation, `ForceTorqueSensor`, additional Genesis rigid-body simulation options,
  a benchmark command with regression gating, and three new robots: **Unitree H1**,
  **UR10e**, and **Allegro Hand**.
- **Multi-robot API:** added `env.articulations[name]` and per-entity routing via
  `SceneEntityCfg.name`, `ActionTermCfg.asset_name`, and `SensorCfg.entity_name`.
- **Public extension API:** added `genelab.extensions` and `genelab.registry.Runnable` as the
  stable extension surface for third-party robots, environments, tasks, and backends.
- **Architecture tooling:** added a required import-linting CI gate and additional guards for
  optional dependencies, CLI help output, and articulation size.

### Changed

- **Breaking:** removed the singular `env.robot`, `env.robot_state`, and `env.articulation`
  accessors, along with name-table convenience accessors such as `joint_names` and
  `default_joint_pos`. Entities should now be accessed by name, for example:
  `env.articulations["robot"].{gs_handle, data, joint_names, …}`.
- Broke the cycle between the RL runner and RL backends.
- Added a shared base manager for term-keyed managers.
- Consolidated small duplicated implementations.
- Decomposed the CLI dispatcher.
- Moved configuration parsing ownership into the domain layer.
- Split task-specific rewards.
- Renamed and colocated vector-environment adapters under the canonical RL vector-env layout.
- Promoted layering checks from advisory linting to a blocking import-linting gate.

### Removed

- Removed deprecated RL wrapper and vector-environment re-export shims. Import vector
  environments from their canonical locations instead.
- Removed unused `RslRlBaseRunnerCfg` fields: `resume`, `load_run`, and `load_checkpoint`.

## [0.1.0] — 2026-05-14

- Added `InteractiveScene` and extracted `Articulation` / `RigidObject` entities from
  `ManagerBasedRlEnv`.
- Split simulation and scene configuration.
- Added the `Sensor[T]` abstraction with contact, body velocity, ray-cast, and terrain
  height sensors.
- Added a mouse-interaction viewer plugin.
- Added rsl_rl-based PPO training and replay for the inverted-pendulum and Unitree G1
  examples.
