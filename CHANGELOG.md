# Changelog

All notable changes to GeneLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is still on a 0.x
trajectory so breaking changes can land in any minor release until the 1.0 stabilisation.

## [Unreleased]

_Nothing yet._

## [0.2.0] — 2026-05-23

Second development release. The M1–M3 product milestones and the §9 architecture
refactor (R0–R7) all landed. Entries are summarised by milestone; see the git history /
`ROADMAP.md` §4 + §9 for per-PR detail.

### Added

- **M1 — reproducibility:** `genelab eval` (deterministic rollout → `eval.json`),
  `genelab export` (TorchScript / ONNX, dependency-free `nn.Module` + `metadata.json`),
  `genelab train --seeds a,b,c [--parallel N]` (multi-seed fan-out), `--eval-every K`
  (in-training EvalCallback + best-model save), the reference-runs doc, `CameraSensor`
  (RGB-D), and five bundled `asset_zoo` robots.
- **M2 — sim2real hardening (M2.1–M2.7):** actuator/joint domain randomization
  (`randomize_joint_stiffness_damping`, `randomize_actuator_deadzone`) + interval-mode DR;
  out-of-limit terminations (`joint_pos/vel_out_of_limit`, `contact_force_limit`); reward
  hard-constraints (`applied_torque_l2`, `joint_vel_limits`, `lin_vel_z_l2`, `base_height_l2`,
  `alive_bonus`); `MlpResidualActuator`; the `ScaledNoise` / `CorrelatedNoise` / `BiasDrift`
  observation-noise models; and the `docs/best-practices/sim2real` deployment recipe.
- **M3 — platform breadth (M3.1–M3.8):** three new `SubTerrainCfg` types + difficulty-ordered
  terrain curriculum (M3.2/M3.3); camera segmentation `CameraSensorCfg.render_segmentation`
  (M3.4); `ForceTorqueSensor` (M3.5); `SimulationCfg` Genesis `RigidOptions` (8 fields, M3.7);
  `genelab benchmark --suite` command + regression gate (M3.8); and three new robots —
  **Unitree H1**, **UR10e**, **Allegro Hand** (M3.1, asset_zoo 5 → 8, covering
  locomotion / arm / dexterous; blobs hosted in `genelab-assets`, md5-pinned).
- **Multi-robot API** (M3.6 / ADR-0012): `env.articulations[name]` + per-entity routing via
  `SceneEntityCfg.name` / `ActionTermCfg.asset_name` / `SensorCfg.entity_name`.
- **Public extension API** `genelab.extensions` (+ `genelab.registry.Runnable`) — the stable
  surface for third-party robots / envs / tasks / backends (ADR-0008).
- Architecture tooling: a required `lint-imports` CI gate (6 contracts / 0 violations,
  ADR-0009); `tests/test_optional_deps.py` (optional-dep boundary), `test_cli_help_snapshots.py`,
  `test_articulation_size.py` (ADR-0010 guard).

### Changed

- **Breaking (M3.6):** removed the singular `env.robot` / `env.robot_state` / `env.articulation`
  and the name-table convenience accessors (`joint_names` / `default_joint_pos` / …). Reach an
  entity by name: `env.articulations["robot"].{gs_handle, data, joint_names, …}`.
- **§9 architecture refactor (R0–R7), all internal / no behaviour change:** broke the
  `rl.runner ↔ rl.backends` cycle (ADR-0001); `BaseTermManager` for the four term-keyed
  managers (ADR-0002); consolidated small duplicates (ADR-0003); decomposed the CLI dispatcher
  (ADR-0004); domain-owned cfg parsing (ADR-0005); split task-specific rewards (ADR-0006);
  renamed/colocated the vec-env adapters to `rl/vecenvs/` (ADR-0007); the layering became a
  blocking import-linter gate (ADR-0009).

### Removed

- The R-phase deprecation shims (old `rl.{rsl_rl,sb3,skrl}_wrapper`, top-level vecenv
  re-exports) — import from the canonical `rl/vecenvs/` locations.
- Dead `RslRlBaseRunnerCfg` fields (`resume` / `load_run` / `load_checkpoint`).

## [0.1.0] — 2026-05-14

- M1 milestone: `InteractiveScene` + `Articulation` / `RigidObject` entities extracted
  from `ManagerBasedRlEnv`; `SimulationCfg` / `InteractiveSceneCfg` split.
- Sensor abstraction (`Sensor[T]`) with contact, body velocity, ray-cast, and terrain
  height sensors.
- Mouse-interaction viewer plugin.
- rsl_rl-based PPO training/replay for the inverted-pendulum and Unitree G1 examples.
