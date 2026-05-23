# Changelog

All notable changes to GeneLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is still on a 0.x
trajectory so breaking changes can land in any minor release until the 1.0 stabilisation.

## [Unreleased]

### Added

- **Multi-robot sensor routing** (ROADMAP M3.6 / ADR-0012, slice **S4**) — `SensorCfg` gains
  `entity_name: str = "robot"`, so each sensor attaches to / reads from a named scene entity.
  - New `genelab.sensor._entity` (`entity_handle` / `entity_state` / `entity_articulation`)
    resolves `env.articulations[entity_name]` with a primary fallback (lives in the sensor
    layer because `mdp._helpers` imports `sensor.contact` — importing it back would cycle).
  - Every sensor (`camera`, `force_torque`, `contact`, `self_contact`, `imu`, `ray_cast`,
    `body_velocity`, `frame_transformer`, `angular_momentum`) now reaches its entity's
    Genesis handle / `RobotState` / articulation — and resolves its link/joint **indices**
    against that entity's name tables — via `entity_name`.

  Backward-compatible: `entity_name` defaults to the primary `"robot"`; single-robot scenes
  are unchanged. Small test-fake updates put `joint_names` on the FT fake articulation
  (mirroring the real entity). Tested in `tests/test_multi_robot.py` (resolver selection +
  a `ForceTorqueSensor` routed to a non-primary entity).

- **Multi-robot write-side term routing** (ROADMAP M3.6 / ADR-0012, slice **S3b**) —
  completes the MDP-layer migration started in S3a. Event, DR and curriculum terms now
  route through `asset_cfg` to the named entity instead of the singular accessors:
  - New `genelab.mdp._helpers.asset_handle(env, asset_cfg)` returns the entity's raw Genesis
    handle (`env.articulations[name].gs_handle`, else primary `env.robot`).
  - `mdp.events` (`reset_root_state_uniform`, `reset_joints_to_default`,
    `reset_joints_by_offset`, `push_by_setting_velocity`) and `mdp.dr`
    (`randomize_joint_stiffness_damping`, `randomize_actuator_deadzone`,
    `body_com_offset` / `body_mass_offset`, `geom_friction`, `encoder_bias`) and
    `mdp.curriculums.terrain_levels_vel` gain an `asset_cfg` (default → primary `"robot"`)
    and reach the entity via `asset_handle` / `asset_articulation` / `asset_state`.

  With S3a + S3b, no `mdp` term reads the singular `env.robot` / `env.robot_state` /
  `env.articulation` / `env.default_joint_pos` / `env.actuators` runtime accessors any more
  (only the `env.joint_names` name table remains, handled by the S6 flip). Backward-compatible
  (managers call terms via `**params`); small test-fake updates put `default_joint_pos` on the
  fake articulation, mirroring the real env. Tested in `tests/test_events.py` + `tests/test_dr.py`.

- **Multi-robot read-side term routing** (ROADMAP M3.6 / ADR-0012, slice **S3a**) — every
  reward / observation / termination term that reads robot state now takes an optional
  `asset_cfg: SceneEntityCfg | None = None` and routes through it, so a multi-robot task can
  point any such term at a specific articulation by name.
  - New `genelab.mdp._helpers.asset_state(env, asset_cfg)` / `asset_articulation(env, asset_cfg)`
    (and the `site_pos_w` / `site_lin_vel_w` foot-site helpers) read the entity named by
    `asset_cfg.name`, defaulting (`None`) to the primary `"robot"`.
  - Migrated `mdp.observations` (base/joint state), `mdp.rewards` (velocity tracking,
    height/orientation, torque/limit penalties, gait/foot rewards), and `mdp.terminations`
    (orientation, height, joint pos/vel limits) off the singular `env.robot_state` /
    `env.joint_*_limits` accessors.

  Backward-compatible: the managers call terms via `**params`, so single-robot tasks and
  examples that pass no `asset_cfg` are unchanged (default → primary). Tested in
  `tests/test_multi_robot.py`; all existing reward/observation/termination suites stay green.
  The write-side terms (`events` / `dr` / `curriculums`, which reach the raw Genesis handle)
  migrate in **S3b**; the singular `env.robot*` accessors are removed in the final flip (S6).

- **Multi-robot action/command routing** (ROADMAP M3.6 / ADR-0012, slice **S2**) — action and
  command terms now honour their (previously-dead) `asset_name`, routing to the named entity:
  - `genelab.mdp._helpers.resolve_articulation(env, name)` and `resolve_robot_state(env, name)`
    return the entity named by `asset_name` (`env.articulations[name]` / its `.data`), falling
    back to the singular primary (`env.articulation` / `env.robot_state`) — so single-robot
    tasks and fake-env tests are unchanged.
  - All four action terms (`JointPositionAction`, `BinaryGripperAction`,
    `ContinuousGripperAction`, `DifferentialIKAction`) match joints, read state and write
    targets against `asset_name`'s entity; the two command terms (`UniformVelocityCommand`,
    `MotionCommand`) read state from it. The write/config path uses `resolve_articulation`, the
    state reads use `resolve_robot_state` — preserving the prior "write via articulation, read
    via robot_state" split exactly.

  Zero behaviour change for single-robot tasks. Tested in `tests/test_multi_robot.py`
  (`resolve_articulation` selection + a `JointPositionAction` routed to a non-primary entity).
  Per ADR-0012, the remaining reward/observation/termination/event term sites + sensors migrate
  in S3/S4.

- **Multi-robot env foundation** (ROADMAP M3.6 / ADR-0012, slice **S1**) — first slice of the
  multi-robot API:
  - `ManagerBasedRlEnvCfg.robots: dict[str, ArticulationCfg]` — when non-empty, the env
    spawns one articulation per entry; when empty it falls back to `{"robot": robot}`, so
    single-robot tasks are unchanged.
  - `env.articulations: dict[str, Articulation]` accessor; the env now holds all entities and
    designates a primary (`"robot"` or the first key) that the singular `env.robot` /
    `env.robot_state` / `env.articulation` accessors alias (those are removed in a later slice
    once terms route by name).
  - `SceneEntityCfg.resolve` is now **entity-aware** — it indexes joint/link names against
    `env.articulations[name]`, activating the previously-dead `SceneEntityCfg.name` field.
    Falls back to the env's primary tables when `articulations` is absent, so existing terms
    and tests are unchanged.

  Zero behaviour change for current single-robot tasks. The actual multi-robot env *build* is
  exercised by the `genesis_runtime`-gated scene test; the resolution / selection logic is
  unit-tested in `tests/test_multi_robot.py`. (Per ADR-0012, the ~127 term-call-site migration
  and removal of the singular accessors land in subsequent slices.)

- **`genelab benchmark` command + suite scaffolding** (ROADMAP M3.8) — runs `eval` across a
  JSON suite of tasks and aggregates the per-task metrics into one report.
  `genelab benchmark --suite suite.json [--out report.json] [--reference ref.json]
  [--tolerance 0.1]`; the suite is a list of `{task, checkpoint, episodes?, seed?, num_envs?}`.
  With `--reference` (a prior report) it flags tasks whose `return_mean` dropped more than
  `--tolerance` and exits non-zero — a regression gate. Orchestration in `genelab.rl.benchmark`
  (`load_suite` / `run_benchmark` / `detect_regressions`), imported function-locally by the CLI
  so `import genelab.cli` stays torch-free. Tested in `tests/test_benchmark.py` (eval mocked).
  (The remaining M3.8 acceptance — ≥8 tasks with real reference numbers + a vision task — needs
  a Genesis runtime + trained checkpoints + hosted assets, blocked in this environment as with M3.1.)

- **Camera segmentation** (ROADMAP M3.4) — `CameraSensorCfg` gains `render_segmentation`
  and `colorize_segmentation` flags; `CameraData` gains a `segmentation` channel. The
  sensor now forwards `segmentation` / `colorize_seg` to Genesis's `camera.render` (the
  segmentation slot was previously discarded) and returns either an `(num_envs, H, W)` int32
  object-index map (level per Genesis `VisOptions.segmentation_level`) or, when colorized,
  an `(num_envs, H, W, 3)` uint8 image. Off by default → no change for existing cameras.
  Tested in `tests/test_sensor.py`. (Depth-derived point clouds — the other half of M3.4 —
  are deferred; Genesis's renderer emits no point-cloud channel.)

- **Terrain curriculum activation** (ROADMAP M3.3) — `TerrainGeneratorCfg.curriculum=True`
  now orders the terrain rows easiest → hardest by a new `SubTerrainCfg.difficulty` field
  (row 0 = lowest difficulty), instead of the previous proportion-weighted random tiling
  (which ignored the flag). Each row uses its single difficulty-ranked sub-terrain type,
  spread evenly when row/terrain counts differ. Pairs with the existing
  `mdp.terrain_levels_vel` env curriculum (which indexes rows by each env's terrain level),
  so promoting an env up a level now means genuinely harder terrain. Note: Genesis keys
  `subterrain_parameters` by *type*, so difficulty must vary across distinct sub-terrain
  types — in-type scaling ("steeper stairs per row") is not expressible. Tested in
  `tests/test_terrain_generator.py`.

- **More sub-terrains** (ROADMAP M3.2) — three new `SubTerrainCfg` types wrapping Genesis
  `parse_terrain` branches: `DiscreteObstaclesCfg` (`discrete_obstacles_terrain` — randomly
  placed rectangles), `SteppingStonesCfg` (`stepping_stones_terrain` — stones + gaps), and
  `FractalCfg` (`fractal_terrain` — multi-octave noise). Exported from `genelab.terrains` +
  `genelab.lab`; tested in `tests/test_terrain_generator.py`. (M3.2's "gaps" has no Genesis
  terrain branch and "mesh import" is the separate `Terrain.height_field` path — both
  deferred.)

- **`SimulationCfg` rigid-solver options** (ROADMAP M3.7) — eight optional fields exposing
  Genesis `RigidOptions`: `enable_self_collision`, `enable_joint_limit`, `max_collision_pairs`
  (contact); `solver_iterations`, `ls_iterations`, `solver_tolerance`, `integrator` (solver);
  `constraint_timeconst` (constraint stiffness/damping). All default `None` ("use Genesis
  default"); `SimulationCfg.rigid_options_kwargs()` maps the set ones to `RigidOptions` kwargs
  and `InteractiveScene` passes `rigid_options` to `gs.Scene` **only when at least one is set**
  — so existing configs are byte-for-byte unchanged. `integrator` is a string resolved to
  `gs.integrator.<name>` in the scene (keeps `configs` Genesis-free, invariant #5). (Genesis
  exposes no continuous-collision-detection / CCD knob, so it is not surfaced.) Tested in
  `tests/test_configs.py`.

- **`ForceTorqueSensor`** (ROADMAP M3.5) — a joint force-torque sensor reporting each
  selected joint's internal reaction force/torque from Genesis `get_dofs_force` (the
  total internal DoF force, distinct from the commanded `applied_torque`). Select joints
  via `joint_names` / `joint_names_expr` (default: all actuated joints); `ForceTorqueData.force`
  is `(num_envs, num_joints)`. Comes with the `mdp.joint_force_torque(env, sensor_name)`
  observation term and a new `Articulation.actuated_dof_ids` accessor (joint→global-DoF
  map). Exported from `genelab.sensor` + `genelab.lab`; tested in `tests/test_sensor.py`.
  (Full 6-axis wrench / fingertip-pressure array deferred, per M3.5.)

- **Sim2Real deployment-recipe doc** (ROADMAP M2.7) —
  `docs/best-practices/sim2real.{en,zh}.md` (How-to Guides → "Harden for Sim2Real"):
  which DR / observation-noise to enable while training (the M2.1/M2.2/M2.6 events +
  models), the hardening terminations/rewards (M2.3/M2.4), the optional learned
  actuator (M2.5), what `genelab export` dumps (TorchScript/ONNX + `metadata.json`),
  and how to align the obs vector / `action_scale` on the hardware side (and what
  not to replicate). **Completes M2.**

- **Observation-noise models** (ROADMAP M2.6) — three new `NoiseCfg` types in
  `genelab.mdp.noise`, re-exported from `genelab.mdp` and `genelab.lab`:
  - `ScaledNoise(n_min, n_max)` — multiplicative scale-factor noise
    (`data × (1 + U(n_min, n_max))`); corruption grows with the signal magnitude.
  - `CorrelatedNoise(std, alpha)` — temporally-correlated AR(1) noise
    (`x_t = α·x_{t-1} + √(1−α²)·N(0, std²)`), stationary std = `std` for any `α`.
  - `BiasDrift(drift_std, max_bias)` — slowly-drifting additive bias (random walk,
    optionally clamped to `±max_bias`).

  `CorrelatedNoise` / `BiasDrift` are **stateful**: a per-element buffer is carried
  across steps (lazily sized to the observation term; the observation manager
  deep-copies the cfg so each term gets its own state). The state is not reset on
  episode boundaries — like a real correlated / drifting sensor error. All three plug
  into the existing `ObservationTermCfg.noise` + `enable_corruption` path with no
  manager change. Tested in `tests/test_sensor.py`.
- **Actuator-level domain randomization** (ROADMAP M2.1):
  - `mdp.dr.randomize_joint_stiffness_damping(env, env_ids, stiffness_range, damping_range)`
    — per-env multiplicative DR on each actuator's PD gains. Force-channel actuators
    (IdealPD/DCMotor/MlpResidual) get a per-env gain *scale* used inside `compute`;
    implicit-PD actuators have their sim-side kp/kv rewritten to `nominal × mult` via
    Genesis.
  - `mdp.dr.randomize_actuator_deadzone(env, env_ids, deadzone_range)` — per-env,
    per-joint torque deadzone (zeroes computed efforts below the half-width; models
    actuator stiction / driver backlash).
  - Plumbing on `ActuatorBase`: per-env `_kp_scale`/`_kv_scale` buffers +
    `set_gain_scale`, a `deadzone` cfg field + `_deadzone` buffer + `set_deadzone` /
    `apply_deadzone` (applied to the computed effort in the articulation control
    loop). Both default to no-ops (scales = 1, deadzone = 0), so **no behaviour
    change** for existing tasks. Per-env gain scaling activates only when `compute`
    runs on the full `num_envs` batch (the production path), leaving flexible-batch
    unit-test calls untouched. New `env.actuators` accessor.

  Scope note (M2.1): the remaining ROADMAP M2.1 items are already covered or
  Genesis-blocked — `push_robot` ≈ existing `mdp.events.push_by_setting_velocity`;
  `randomize_imu_bias` ≈ the IMU sensor's per-env `bias_range_*` resampled on reset;
  `randomize_restitution` / `randomize_gravity` have no Genesis setter
  (`RigidEntity` exposes friction only; `Scene` gravity is construction-time).
- **`MlpResidualActuator`** (ROADMAP M2.5) — a learned-residual actuator:
  `effort = clamp(DCMotor_base(q, q̇, q*) + scale · net([q*−q, q̇]), ±effort_budget)`.
  The residual network is a TorchScript module loaded from
  `MlpResidualActuatorCfg.network_file` (GeneLab only loads + runs it; training the
  weights lives downstream, per M2.5). Single-step, per-joint contract: net input
  `(…, 2)` = `[pos_error, joint_vel]`, output `(…, 1)` = residual torque (a plain
  `nn.Linear(2, …)` first layer satisfies it). With no `network_file` it degrades to
  a pure `DCMotorActuator`. Exported from `genelab.actuator` and `genelab.lab`;
  tested in `tests/test_actuator.py`.
- **`contact_force_limit(env, sensor_name, max_force)` termination** (ROADMAP
  M2.3, final slice) — trips when any link tracked by the named `ContactSensor`
  has a net contact-force magnitude (`force_norm`) above `max_force`. A safety
  termination for impact spikes (e.g. base/knee ground slams). **Completes M2.3 /
  M2.4** — all out-of-limit terminations + reward hard-constraints are now in
  `genelab.mdp`. Unit-tested in `tests/test_terminations.py`.
- **MDP hard-constraint terms** (ROADMAP M2.3 / M2.4, first slice): four new
  reusable functions in `genelab.mdp`, re-exported from the package namespace —
  - reward `lin_vel_z_l2(env)` — penalize vertical base velocity (`v_z²`),
  - reward `base_height_l2(env, target_height)` — squared base-height deviation
    (flat-ground variant),
  - reward `alive_bonus(env)` — constant `+1` per alive step,
  - termination `joint_pos_out_of_limit(env)` — trips when any actuated joint
    leaves its position limits (reuses `env.joint_pos_limits`).

  Each has unit tests (`tests/test_rewards.py`, new `tests/test_terminations.py`).
- **Velocity-limit + applied-torque MDP terms** (ROADMAP M2.3 / M2.4, second
  slice) — plus the data plumbing they need:
  - `RobotState.applied_torque` — realized actuator torque, refreshed each step from
    Genesis `get_dofs_control_force` (zero on fake envs / platforms without it).
  - `ArticulationCfg.joint_vel_limit: float | None` — optional uniform soft joint
    velocity limit (rad/s); Genesis exposes no per-joint velocity limit, so it's
    user-declared. `None` → `+∞` (terms inert). New `Articulation.joint_vel_limits`
    / `env.joint_vel_limits` accessors mirror the existing `joint_pos_limits` ones.
  - reward `applied_torque_l2(env)` — `Σ τ²` over actuated joints.
  - reward `joint_vel_limits(env, soft_ratio=1.0)` — `Σ max(0, |q̇| − ratio·limit)`.
  - termination `joint_vel_out_of_limit(env)` — trips when any joint exceeds its
    velocity limit.

  Opt-in and inert until a task sets `joint_vel_limit` — no behaviour change for
  existing tasks. The contact-force term (`contact_force_limit`) remains deferred
  (sensor-coupled). Unit-tested in `tests/test_rewards.py` + `tests/test_terminations.py`.
- **`tests/test_articulation_size.py`** — size guard for `entity/articulation.py`,
  fulfilling ADR-0010 (defer the entity/articulation split) §Risks R10.1 /
  Validation. The split stays deferred; this is the recorded soft check that the
  file isn't accreting unnoticed past the point a split would be cheap: silent
  pass ≤700 LoC, a `UserWarning` (prompting a revisit of the ADR's trigger
  criteria) at 700–1000, hard failure >1000. The file is 528 LoC today.
- **Public extension API** `genelab.extensions` (ADR-0008 / ROADMAP §9 R7): a
  single import path for the four extension kinds —
  `from genelab.extensions import register_robot, register_env, register_task,
  register_backend, ROBOTS, ENVS, TASKS, Backend, Runnable`. The symbols are
  re-exports of the existing `genelab.registry` and `genelab.rl.backends`
  implementations (which keep working unchanged); `genelab.extensions` is the
  canonical, stable surface for third-party packages. `register_backend` — the
  one previously-undocumented registration function — is now discoverable here.
- `genelab.registry.Runnable` — the (now public) Protocol every `TASKS` value
  must satisfy after instantiation (`cfg`, `play()`, `train()`). Promoted from
  the former private `cli/__init__.py:_RunnableTask`; the CLI imports it from
  `registry`. (The transitional `_RunnableTask = Runnable` alias was removed
  within this same cycle — see Removed, #110.)

### Removed

- **Dropped the R-phase deprecation shims** (backward-compat for the relocated
  modules from R6/R7 is no longer maintained). The following old import paths are
  **gone** — import from the canonical locations instead:
  - `genelab.rl.rsl_rl_wrapper` / `genelab.rl.sb3_wrapper` / `genelab.rl.skrl_wrapper`
    → `genelab.rl.vecenvs.{rsl_rl,sb3,skrl}` (the vecenv adapters, ADR-0007 / R6).
  - `genelab.rl.RslRlVecEnvWrapper` (top-level re-export via `__getattr__`)
    → `genelab.rl.vecenvs.rsl_rl.RslRlVecEnvWrapper`.
  - `genelab.rl.distributed` → `genelab.utils.distributed` (the torchrun helpers,
    ADR-0009 / R7.3b).
  - `genelab.cli._eval.eval_task` (CLI re-export) → `genelab.rl.eval_task.eval_task`
    (ADR-0009 / R7.3a).
  - `genelab.cli._RunnableTask` (private alias) → `genelab.registry.Runnable`
    (ADR-0008 / R7.1).

  `tests/test_deprecated_imports.py` (which asserted the shims warned + re-exported)
  is removed with them. No behaviour change for the canonical surface; the three
  internal/test references that still used an old path were repointed.

### Changed

- **`MetricsManager` and `CurriculumManager` now subclass `BaseTermManager`**
  (ADR-0002 addendum / ROADMAP §9 R2.5 dedup leftover; no behaviour change). Both
  carried verbatim copies of the term-registration loop (`deepcopy` cfg →
  `instantiate_class_term` per term → parallel `_term_names` / `_term_cfgs`) that
  R2.5 already consolidated into `BaseTermManager` for the reward / termination
  managers. `CurriculumManager.__init__` was byte-identical to the base (it now
  inherits it outright); `MetricsManager` moved its `_episode_sums` /
  `_step_count` allocation into the `_post_init` hook. The duplicated `num_envs` /
  `device` / `active_terms` properties are dropped in favour of the base's. Public
  constructor signatures (`(cfg, env)`) are unchanged.
  `tests/test_manager_init_order.py` gained Metrics + Curriculum init-order
  assertions (committed and verified green *before* the refactor, per the R2.5
  gate-test pattern).
- **Architecture lint is now a required CI gate** (ADR-0009 / ROADMAP §9 R7.3d —
  completes the R0–R7 refactor). The `lint-imports` step dropped its
  `continue-on-error`, so any PR that introduces a cross-layer import now fails CI.
  The baseline is clean: **6 contracts, 0 violations**. To get there, the
  monolithic `layers` contract (which treated the 11 domain packages as mutually
  independent and so wrongly forbade legitimate intra-domain imports like
  `envs → scene`, `mdp → managers`, `entity → actuator`) was replaced by
  directional `forbidden` contracts that enforce the band ordering
  (`cli > rl > domain > config-band > utils`) while leaving intra-band imports
  free: added "rl is below cli" and "Infrastructure modules do not import up". A
  new `tests/test_importlinter_configured.py` asserts the contract set stays
  present so the gate can't be silently disabled by deleting the config.
- **Internal restructuring** (no behaviour change): `PROJECT_ROOT` / `CACHE_DIR`
  moved from `genelab.cache` to a new `genelab.utils.paths` module (ADR-0009 /
  R7.3d), so `utils.download` resolves the asset cache without importing up into
  `genelab.cache` (the last infra→config-band edge). `genelab.cache` re-exports
  them, so `from genelab.cache import CACHE_DIR` and `ensure_project_cache` are
  unchanged.

- **Architecture lint** (no code change): split the "Domain modules are below
  cli / rl / utils.download" importlinter contract into two — "Domain modules are
  below cli / rl" (all domain packages) and "Domain (except asset_zoo) does not
  import utils.download" (ADR-0009 / ROADMAP §9 R7.3c). `asset_zoo` is the asset
  catalog, so fetching assets via `utils.download` (`fetch_asset(AssetSpec(...))`
  for URDF/MJCF/motion files) is a legitimate downward `domain → utils` import,
  not a layering violation; the split states that intent while keeping the
  "term logic must not download" guard for the other ten domain packages. With
  R7.3a + R7.3b already clearing the `domain → rl` leaks, both forbidden contracts
  now pass — the importlinter baseline is **4 kept / 1 broken** (only the
  "Top-down layering" contract remains, addressed in R7.3d before the blocking
  flip).

- **Internal restructuring** (no behaviour change): the torchrun helpers moved
  from `genelab.rl.distributed` to `genelab.utils.distributed` (ADR-0009 /
  ROADMAP §9 R7.3b). They are a generic environment/torchrun utility (only `os` +
  a deferred `torch`; no RL-specific content), and a domain module
  (`scene.interactive_scene`) needs `pin_cuda_device` — which `rl` may not sit
  below. Moving the module to the `utils` band clears the
  `scene → rl.distributed` layering violation (and the `envs → scene → rl` chain
  through it); after R7.3a + R7.3b there are **no `domain → rl` violations left**.
  All internal callers (the three backends, `rl/profiler.py`, `scene`, and a CLI
  test) now import from `genelab.utils.distributed`. Module moved verbatim. The old
  `genelab.rl.distributed` path shipped briefly as a `DeprecationWarning` re-export
  shim and was removed within this same cycle (see Removed, #110).
- **Internal restructuring** (no behaviour change): `eval_task` moved from
  `cli/_eval.py` to a new `genelab.rl.eval_task` module (ADR-0009 / ROADMAP §9
  R7.3). Its body is backend-agnostic eval orchestration whose dependencies all
  live in the `rl` / config bands (`cache`, `registry`, `rl.evaluator`,
  `rl.backends`, `rl.runner`) — it imported nothing from `cli`. Both callers (the
  `genelab eval` command and the in-training `EvalCallback`) now reach it in the
  `rl` layer, which removes the `rl.eval_callback → cli._eval` layering violation
  (the last `rl → cli` import). `cli/_eval.py` re-exported `eval_task` as a shim
  that was removed within this same cycle (see Removed, #110); callers import
  `genelab.rl.eval_task.eval_task` directly. The `rl.runner` import is
  function-local in the moved function to keep the import graph acyclic. Body is
  byte-identical to before apart from that import.

- **Internal restructuring** (no behaviour change): the three vec-env adapters
  moved verbatim from `rl/<lib>_wrapper.py` into `rl/vecenvs/<lib>.py`, so each
  adapter sits next to its same-named trainer under `rl/backends/<lib>.py` and
  the file tree expresses the per-library pairing (ADR-0007). The shared
  `attach_optional_base` helper (R2.2) relocated alongside them to
  `rl/vecenvs/_attach_base.py` — its R2.2 deferred home. All internal callers
  (the three backends, `skrl_models.py`, and the RL pipeline tests) import the
  canonical `rl/vecenvs/<lib>` paths. The old `rl/<lib>_wrapper.py` paths and the
  top-level `genelab.rl.RslRlVecEnvWrapper` re-export shipped briefly as
  `DeprecationWarning` shims and were **removed within this same unreleased cycle**
  (see Removed, #110), so they net to absent in the release. `tests/test_optional_deps.py`
  now also covers the three `rl/vecenvs/<lib>` modules. Completes ADR-0007 (R6).

- **Internal refactor** (no behaviour change): the three jaccard-1.000
  motion-tracking body-error rewards are now thin wrappers over a shared
  `motion_body_error_exp(env, command_name, std, body_names=None, *, quantity)`
  factory in `mdp/motion_tracking.py` (ADR-0006 / ROADMAP §9 PR R5.2). `quantity`
  (`"pos"` / `"lin_vel"` / `"ang_vel"`) selects the `(reference, robot)` attribute
  pair on the `MotionCommand`; the public names
  (`motion_relative_body_position_error_exp`,
  `motion_global_body_linear_velocity_error_exp`,
  `motion_global_body_angular_velocity_error_exp`) keep their exact signatures and
  `__name__` (thin `def` wrappers, not `functools.partial`, so reward-term logging
  is unaffected). The factory is also exported from the `genelab.mdp` namespace for
  direct use. The orientation (geodesic) and anchor rewards are left as-is — they
  are not part of the jaccard-1.000 duplication. New `tests/test_motion_tracking_equivalence.py`
  pins the pre-refactor implementations and asserts the factory/wrappers reproduce
  them bit-for-bit (`torch.equal`), with and without the `body_names` filter.
  Completes ADR-0006 (R5).

- **Internal restructuring** (no behaviour change): the motion-imitation
  reward family moved out of `mdp/rewards.py` into a new
  `mdp/motion_tracking.py` (ADR-0006 / ROADMAP §9 PR R5.1), so the generic
  reward library stays a coherent "any task may use this" surface. The whole
  "motion imitation" section relocated **verbatim** — six public functions
  (`motion_global_anchor_position_error_exp`,
  `motion_global_anchor_orientation_error_exp`,
  `motion_relative_body_position_error_exp`,
  `motion_relative_body_orientation_error_exp`,
  `motion_global_body_linear_velocity_error_exp`,
  `motion_global_body_angular_velocity_error_exp`) plus the shared private
  helpers `_motion_command` / `_body_index_filter`. `mdp/rewards.py` re-exports
  the six (PEP-484 `as` idiom), so both `genelab.mdp.motion_*` (the package
  namespace all consumers use) and `genelab.mdp.rewards.motion_*` keep
  resolving — `mdp/__init__.py`, the Unitree G1 example, and the tests are
  unchanged. `mdp/rewards.py` shrinks 554 → 461 LoC (also dropped the now-unused
  `cast` / `MotionCommand` / `quat_error_magnitude` imports). **ADR variance:**
  ADR-0006 named only the three jaccard-1.000 functions; the "motion imitation"
  section had since grown to six + the two helpers, so R5.1 moved the whole
  coherent block (keeps the shared helpers with their only users). The
  parameterized `motion_body_error_exp` factory + numerical-equivalence test
  remain for R5.2.
- **Internal restructuring** (no behaviour change): the play / train dispatch
  moved out of `cli/__init__.py` into a new `cli/_dispatch.py` submodule,
  completing the CLI dispatcher decomposition (ADR-0004). `_dispatch_play` and
  `_dispatch_train` relocated verbatim and are re-exported from `genelab.cli`
  so the `play` / `train` Typer callbacks keep calling them unchanged. The
  profiler-kwarg coercion (`_coerce_prof_kwargs` + the private
  `_parse_bool` / `_parse_int` / `_parse_path` helpers) and the `_AGENT_KINDS`
  set moved alongside them — the two dispatch functions are their only users,
  so co-locating keeps `_dispatch.py` a self-contained leaf and avoids a
  runtime `cli → _dispatch → cli` import cycle (ADR-0004 had tentatively kept
  `_coerce_prof_kwargs` in `__init__.py`; the cycle its R4.2 risk row
  anticipated forced the co-location — recorded as an ADR variance). The
  `_RunnableTask` type hint is a `TYPE_CHECKING`-only forward reference. Three
  imports orphaned by the move (`os`, `typing.Any`, `pick_agent_kind`) were
  dropped from `__init__.py`. Test monkeypatch targets were repointed to the
  new owning modules: the four `_relaunch_under_torchrun` `os.execvp` patches
  to `genelab.cli._distributed.os.execvp`, and the `_patch_picker` helper now
  also patches the `cli._dispatch` consumer site (each importing module holds
  its own binding). `cli/__init__.py` shrinks 775 → 645 LoC. All three modules
  ADR-0004 named are now extracted; the ADR's ≤400-LoC target is not reached
  by this slice (the residue — 10 Typer command callbacks, `_configured_task`
  / `_resolve_task`, the override helpers, `_RunnableTask`, and help text — is
  what ADR-0004 deliberately kept in `__init__.py`), so reaching ≤400 is left
  to a separate follow-up. `--help` text byte-identical (R0.1 snapshot gate
  green); importlinter baseline unchanged at 2 kept / 2 broken (the move stays
  within the `cli` package). Lands as ROADMAP §9 PR R4.3 — completes Phase R4
  / ADR-0004.
- **Internal restructuring** (no behaviour change): the multi-seed train
  orchestration moved out of `cli/__init__.py` into a new
  `cli/_multi_seed.py` submodule. Four functions relocated verbatim —
  `_dispatch_multi_seed_train`, `_parse_seed_list`,
  `_resolve_multi_seed_parent`, `_strip_multi_seed_flags` (plus the
  `_STRIPPABLE_MULTI_SEED_FLAGS` constant) — and are re-exported from
  `genelab.cli` so existing imports (incl. `tests/test_multi_seed_cli.py`)
  keep working unchanged. The new module imports its argv-strip helpers
  (`_extract_log_dir_flag`, `_strip_flag_value_pairs`) from
  `cli/_distributed.py`; the `_RunnableTask` type hint is a
  `TYPE_CHECKING`-only forward reference (no runtime import cycle).
  `cli/__init__.py` shrinks 900 → 775 LoC (the now-unused `import sys` is
  dropped). Four `_relaunch_under_torchrun` tests had their
  `monkeypatch.setattr` target corrected from the stale `genelab.cli.sys.argv`
  to `genelab.cli._distributed.sys.argv` — that function moved to
  `_distributed.py` in R4.1 and no longer relies on `sys` living in
  `cli/__init__.py`. All `genelab` `--help` text is byte-identical (R0.1
  snapshot gate green); the importlinter baseline is unchanged (the move
  stays within the `cli` package). Lands as ROADMAP §9 PR R4.2 — second of
  three sub-PRs in ADR-0004 (CLI dispatcher decomposition).
- **Internal restructuring** (no behaviour change): the distributed
  (multi-GPU) training plumbing moved out of `cli/__init__.py` into a new
  `cli/_distributed.py` submodule. Six functions relocated verbatim —
  `_relaunch_under_torchrun`, `_resolve_per_rank_num_envs`,
  `_strip_distributed_flags`, `_strip_flag_value_pairs`,
  `_extract_log_dir_flag`, `_has_log_dir_flag` — and are re-exported from
  `genelab.cli` so existing imports (incl. `tests/test_cli.py` and
  `tests/test_multi_seed_cli.py`) keep working unchanged. `cli/__init__.py`
  shrinks 1,020 → 900 LoC. All `genelab` `--help` text is byte-identical
  (R0.1 snapshot gate green); the importlinter baseline is unchanged (the
  move stays within the `cli` package). Lands as ROADMAP §9 PR R4.1 — first
  of three sub-PRs in ADR-0004 (CLI dispatcher decomposition).
- **Internal restructuring** (no behaviour change): the play-mode
  shortcut-retargeting key list moved off the private
  `cli/__init__.py:_PLAY_RETARGETED_KEYS` constant onto
  `SimulationCfg.play_retargeted_keys()` (a static method on the domain
  config in `genelab.configs`). The CLI's `env.` → `play_env.` retarget
  loop now calls the method; the set of play-retargetable simulation
  override paths (`env.simulation.{vis,gpu,steps,dt}`) lives next to the
  `SimulationCfg` fields the `--vis` / `--gpu` / `--steps` / `--dt`
  shortcuts target. `genelab play --help` is unchanged (R0.1 snapshot
  gate green); `configs.py` stays torch-free at import (invariant #5).
  Lands as ROADMAP §9 PR R3.2 — completes ADR-0005 (R3).
- **Internal restructuring** (no behaviour change): `--eval-*` runner-arg
  parsing for in-training eval moved from `cli/__init__.py:_build_eval_callback`
  onto the domain config as `EvalCallbackCfg.from_args(runner_args) ->
  EvalCallbackCfg | None` (in `genelab.rl.eval_callback`). The CLI dispatcher
  now forwards the raw flag dict; the config owns parsing its own args. Parse
  behaviour is byte-identical (`--eval-every` unset → `None`; otherwise an
  enabled cfg with int-coerced `--eval-episodes` / `--eval-num-envs` /
  `--eval-seed`, defaulting to 10 / None / 0). `genelab train --help` is
  unchanged (R0.1 snapshot gate stays green). Lands as ROADMAP §9 PR R3.1
  (first of two sub-PRs in ADR-0005). Note: R3.1 does **not** change the
  importlinter baseline — the separate `rl.eval_callback -> cli._eval`
  layering violation (the eval-callback loop calling `eval_task`) is a
  distinct concern tracked for a follow-up slice.
- **Internal dedup** (no public API change): `RewardManager` and
  `TerminationManager` now subclass a new
  `genelab.managers._base.BaseTermManager[TCfg]` (generic in the
  term-cfg type). The shared `__init__` body — deepcopy cfg, build the
  parallel `_term_names` / `_term_cfgs` lists, run
  `instantiate_class_term` per term — and the `num_envs` / `device` /
  `active_terms` properties move to the base (jaccard 0.953 between the
  two former `__init__` methods). Each subclass keeps only its
  domain-specific buffer allocation, now in a `_post_init` hook the
  base calls at the **end** of `__init__` (after term registration) —
  preserving the buffer-allocation timing that `RewardManager.reset` /
  `TerminationManager.reset` depend on. A new
  `tests/test_manager_init_order.py` gate (added in the preceding
  commit, passes pre- and post-refactor) locks this init-order
  invariant. Lands as ROADMAP §9 PR R2.5 — the fifth and final
  small-abstraction sub-slice in ADR-0003 (with the `_post_init`
  template-method shape from ADR-0002); completes Phase R2.
- **Internal dedup** (no public API change): the shared PD-gain-write
  body of `IdealPDActuator.initialize` and `ImplicitPDActuator.initialize`
  (jaccard 0.969 — only the kp/kv tensor source varied: `zeros` for
  ideal, `self._stiffness`/`self._damping` for implicit) moves into a
  new `ActuatorBase._write_pd_gains(gs_handle, *, kp_values, kv_values)`
  helper, alongside the existing `_write_force_range` / `_write_armature`
  / `_write_friction` helpers. Each subclass `initialize` shrinks to
  `super().initialize(gs_handle)` + one helper call (5 lines for ideal,
  4 for implicit). `DCMotorActuator` continues to inherit
  `IdealPDActuator.initialize` unchanged. ADR-0003 names this
  `_initialize_pd_common` but `_write_pd_gains` was chosen for naming
  consistency with the sibling `_write_*` helpers on the base class.
  Lands as ROADMAP §9 PR R2.4 (fourth of five sub-slices in ADR-0003).
- **Internal dedup** (no public API change): the regex joint-name →
  indices code in `BinaryGripperAction.__init__` and
  `ContinuousGripperAction.__init__` (jaccard 1.000 between the two
  bodies — only the error-message class name differed) moves to a new
  private helper `genelab.mdp.actions._joint_match.match_joints`. Each
  gripper now calls `match_joints(cfg.joint_names, env.joint_names)` and
  raises its own term-specific `ValueError` when zero joints match.
  Helper accepts `Sequence[str]` so both the cfg's `tuple[str, ...]`
  and `list[str]` callers work. Tested via the gripper-using paths in
  `tests/test_franka_pick_and_place_examples.py` and `test_ee_delta_ik.py`.
  Lands as ROADMAP §9 PR R2.3 (third of five sub-slices in ADR-0003).
- **Internal dedup** (no public API change): the three
  `_attach_{rsl_rl,skrl,sb3}_base` helpers at the bottom of
  `genelab.rl.{rsl_rl,skrl,sb3}_wrapper` — verbatim copies of each
  other (jaccard 0.984–1.000) that conditionally re-base each wrapper
  on the matching upstream library's vec-env class — collapse into a
  single `genelab.rl._attach_base.attach_optional_base` helper. Each
  wrapper now calls it once with `base_module` / `base_attr` /
  `wrapper_name` / `caller_globals` parameters. The wrappers still
  subclass their optional bases at module load (`RslRlVecEnvWrapper`
  → `rsl_rl.env.VecEnv`, `GenelabSkrlWrapper` →
  `skrl.envs.wrappers.torch.Wrapper`, `GenelabSb3VecEnv` →
  `stable_baselines3.common.vec_env.VecEnv`) when those libraries are
  installed, and degrade to bare object subclasses otherwise; verified
  by `tests/test_optional_deps.py`. Net code reduction: −27 lines.
  Lands as ROADMAP §9 PR R2.2 (second of five sub-slices in ADR-0003).
  Note: ADR-0003 names the new module `rl/vecenvs/_attach_base`; for
  now it lives flat at `rl/_attach_base` because `rl/vecenvs/` is
  created in R6 (ADR-0007). R6 will relocate it.
- **Internal dedup** (no public API change): the `ON_POLICY_ALGORITHMS`
  and `OFF_POLICY_ALGORITHMS` frozensets — previously defined verbatim
  in both `genelab.rl.sb3_config` and `genelab.rl.skrl_config` — moved
  to a new shared module `genelab.rl._algorithm_taxonomy`. Both configs
  re-export the constants so any existing
  `from genelab.rl.sb3_config import ON_POLICY_ALGORITHMS` (or the skrl
  equivalent) keeps working unchanged. Adding a new on/off-policy
  algorithm symbol now means editing exactly one file. Lands as ROADMAP
  §9 PR R2.1, the first of five small-abstraction sub-slices in ADR-0003.
- **Internal restructuring** (no public API change): the nine helpers
  shared by every RL backend (`build_bridges`, `build_env`,
  `close_bridges`, `make_random_policy`, `make_zero_policy`,
  `resolve_env_cfg`, `resolve_log_dir`, `run_play_loop`,
  `save_run_params`) moved from `genelab.rl.runner` into a new private
  module `genelab.rl._helpers`. `rl/runner.py` re-exports the same
  names so external callers using
  `from genelab.rl.runner import build_env, resolve_env_cfg, …` are
  unaffected. The motivation is breaking the static import cycle
  documented in ADR-0001: each concrete backend
  (`rl.backends.{rsl_rl, skrl, sb3}`) used to import the helpers back
  from `rl.runner`, forming a cycle that was held together only by
  the lazy `_ensure_loaded` workaround in `rl/backends/__init__.py`.
  After R1, backends import directly from `rl._helpers`, the cycle is
  gone, and the new `tests/test_no_static_cycle.py` (via `grimp`)
  prevents regression. The R0.3 `import-linter` baseline's
  `rl.backends does not import rl.runner` contract flips from broken
  to kept (3 violations removed; 21 of the original 24 remain).
  Lands as ROADMAP §9 PR R1.

### Added

- `[tool.importlinter]` in `pyproject.toml` + non-blocking
  `Architecture lint (import-linter)` step in `.github/workflows/ci.yml`
  (lint job). Four layering contracts derived from ADR-0009:
  (1) domain modules are below `cli` / `rl` / `utils.download`,
  (2) `rl.backends` does not import `rl.runner` (will be fixed by R1 / ADR-0001),
  (3) top-down layering `cli > rl > domain > utils`,
  (4) the three RL backends are independent of each other.
  `exclude_type_checking_imports = true` filters out the deliberate
  `if TYPE_CHECKING: from envs.manager_based_rl_env import ManagerBasedRlEnv`
  type-hint pattern used across `mdp/` / `managers/` / `sensor/` — this is
  the manager-based MDP API contract, not a layering violation. Baseline
  on `dev`: 1 contract kept, 3 broken (24 distinct cross-layer imports
  in 122 files / 266 dependencies). The CI step runs with
  `continue-on-error: true` so the broken baseline does not block merges;
  R7 (ADR-0009 §"R7 — flip to blocking") will convert it to a required
  check once R1–R6 have trimmed the violation list. Lands as ROADMAP §9
  PR R0.3 — completes Phase R0.
- `tests/test_optional_deps.py` — runtime guard for invariant #1
  (`import genelab.rl` must succeed without `rsl_rl` / `skrl` /
  `stable_baselines3` / `tensordict` installed). Each of the four
  load-bearing entry points (`genelab.rl`, `genelab.rl.backends.rsl_rl`,
  `genelab.rl.backends.skrl`, `genelab.rl.backends.sb3`) is imported
  in a fresh subprocess with the four optional libs poisoned via
  `sys.modules[name] = None`; a non-zero exit code surfaces any
  top-level import that should have been function-local. Pairs with
  the R7 importlinter contract (ADR-0009) which catches the same class
  of regression statically. Lands as ROADMAP §9 PR R0.2.
- `tests/test_cli_help_snapshots.py` + `tests/snapshots/help-*.txt` —
  frozen baseline of every Typer command's `--help` output (root, cache,
  prof, list, info, play, eval, export, train, project, project new). The
  test runs the CLI in a deterministic subprocess (`NO_COLOR=1`,
  `TERM=dumb`, `COLUMNS=100`) and asserts byte-equality against the
  captured snapshot so the upcoming CLI decomposition (ROADMAP §9 Phase
  R4) and domain-owned-parsing refactor (Phase R3) can prove they are
  structural moves rather than behavioural edits. Intentional `--help`
  changes regenerate via `UPDATE_SNAPSHOTS=1 pytest
  tests/test_cli_help_snapshots.py`. Lands as ROADMAP §9 PR R0.1 — the
  smallest gate of the refactor chain.
- `genelab train TASK --seeds 1,2,3 --parallel N` — fan-out multi-seed
  training. Each comma-separated seed is launched as an independent
  `genelab train` subprocess (concurrency capped by `--parallel`), with a
  per-child `--seed S` and `--log_dir <parent>/seed_S` so all seeds of one
  launch land under a shared parent directory. The parent defaults to
  `logs/multi-seed/<task_id>/<timestamp>/` and can be overridden with
  `--log_dir`. Independent from `--gpus N` (distributed within one run);
  the two are orthogonal axes.
- `docs/best-practices/reference-runs.{en,zh}.md` — bilingual
  reproduction protocol for the 5 bundled tasks × 3 seeds. The protocol is
  final (commands, log layout, methodology); the reference numbers
  themselves are TBD and tracked under ROADMAP M1.7. Populated PRs land
  the numbers + TensorBoard curves once a stable Genesis pin is chosen.
- `genelab eval TASK CHECKPOINT` — deterministic rollout that writes a JSON
  summary (`return_mean`/`return_std`, `length_mean`, optional `success_rate`,
  `wall_clock_seconds`) in the ROADMAP §M1.1 schema. Backend-agnostic: routes
  through the new `Backend.make_inference_setup` method so `rsl_rl` / `skrl` /
  `sb3` all share the same `run_evaluation` rollout core. Tasks opt into
  `success_rate` by publishing `extras["is_success"]` per-env from their
  `ManagerBasedRlEnv.step` (gymnasium convention); absent → JSON `null`.
- `genelab train --eval-every K` — periodic in-training eval and best-model
  selection. Each chunk of `K` iters ends with a deterministic eval on the
  newest checkpoint; the eval payload is written to
  `<log_dir>/best_model_meta.json` and the checkpoint is copied to
  `<log_dir>/best_model.<ext>` (`.pt` for `rsl_rl`/`skrl`, `.zip` for `sb3`)
  whenever `return_mean` improves. New flags: `--eval-every`,
  `--eval-episodes`, `--eval-num-envs`, `--eval-seed`.
- `genelab export TASK CHECKPOINT --format {torchscript,onnx}` — backend-agnostic
  policy export. The actor sub-network is wrapped in `ExportedPolicy`, which
  bakes per-term obs `scale` / `clip` into a single `forward(raw_obs) ->
  actions` pass; deployment requires only `torch` (TorchScript) or an ONNX
  runtime, with no `rsl_rl` / `skrl` / `stable_baselines3` import at inference.
  A sibling `<output>.metadata.json` records obs group dims, term-level
  normalization, action dim/range, and provenance.
- `InferenceSetup` (in `genelab.rl.backends.base`) and
  `Backend.make_inference_setup(ctx)` — shared abstraction the eval / export /
  play paths now use to load a checkpoint and hand back the actor module,
  policy callable, and eval-friendly env adapter as one bundle.
- `docs/concepts/eval-and-export.{en,zh}.md` — bilingual concept page covering
  the new CLIs, the success-rate convention, off-policy caveats, deployment-
  side load + forward snippets, and known limitations (no dict-obs export, no
  recurrent policies yet).
- `ROADMAP.md` — tracking document for the next 2–3 release cycles. Defines
  GeneLab's positioning (Genesis-backed manager-based RL scaffold, Isaac Lab
  API shape, multi-backend), seven design principles, three milestones (M1
  research reproducibility, M2 sim2real hardening, M3 platform breadth), and
  a contributor on-ramp.

### Changed

- `joint_acc_l2` (in `genelab.mdp.rewards`) now emits a one-shot
  `warnings.warn` announcing that it returns 0 — the function name implies
  joint-acceleration tracking, but a proper implementation needs prior-step
  `joint_vel` history (tracked under ROADMAP M2). Closes the §3 P4 violation
  flagged in the ROADMAP audit (M1.6).

### Removed

- `RslRlBaseRunnerCfg.resume`, `.load_run`, and `.load_checkpoint` — all three
  fields were dead code (the actual resume path comes from `--checkpoint` /
  `TrainContext.resume_from`). Closes the §3 P3 violation flagged in the
  ROADMAP audit (M1.5).

### Fixed

- Slow-motion playback during `genelab play` with non-trivial `decimation`: the
  Genesis viewer's `max_FPS` rate-limit was applied on every physics tick, so
  e.g. `decimation=10` produced ~8× slow-motion. `ManagerBasedRlEnv.step` now
  refreshes the viewer only on the final physics tick of the decimation loop;
  `InteractiveScene.step` gained an `update_visualizer` kwarg to support that.
- Video save extension: pressing `R` twice in the Genesis viewer to save a
  recorded video, then typing a bare filename in the SaveAs dialog, used to
  write the `.mp4` content to a `.png` path (upstream pyrender hard-codes
  `defaultextension=".png"`). `InteractiveScene._build` now applies a one-time,
  class-level monkey-patch on `genesis.ext.pyrender.viewer.Viewer._get_save_filename`
  to coerce the returned extension to the requested one when exactly one
  extension was offered. Genesis itself is unchanged.

### Added

- `SimulationCfg.render_fps: int | None = 60`: viewer FPS cap decoupled from
  the physics rate (`1/dt`). Forwarded to `gs.options.ViewerOptions(max_FPS=...)`
  when `vis=True`; headless training paths are untouched. Override via the
  existing dotted-path grammar (`env.simulation.render_fps=...`).

- `docs/concepts/rl-runner.{en,zh}.md`: bilingual concept page for `genelab.rl`
  (`train_task` / `play_task` signatures, `RslRlOnPolicyRunnerCfg` field
  layout, `AgentKind` Literal, `RslRlVecEnvWrapper` adapter, torchrun-based
  `--gpus N` distributed-training relaunch, `maybe_profile` flag/env-var
  matrix, failure modes).
- `docs/concepts/mdp.{en,zh}.md`: bilingual reference for the reusable MDP term
  library (`genelab.mdp.{actions,commands,observations,rewards,terminations,
  events,curriculums,noise}`). Every re-exported function / class is tabulated
  with its required params, output shape, and a minimal wiring example, plus
  warnings on the `joint_acc_l2` / `feet_air_time` placeholder rewards.
- `docs/concepts/managers.{en,zh}.md`, `docs/cli/list-info.{en,zh}.md`: bilingual
  manager-system concept page (seven managers + `ManagerTermBaseCfg` + term
  lifecycle + `EventMode` semantics + per-manager wiring examples) and CLI
  discovery page (`list KIND` enumeration, `info NAME` panel anatomy, override
  path lookup workflow, simulation flag shortcuts).
- `examples/genelab_showcase/`: six play-only TaskCfgs that exercise every M1–M4
  building block (sensors / ray-cast patterns / contact + air-time / 5 sub-terrains /
  `terrain_levels_vel` curriculum / `IdealPDActuator` arm). Each task dumps
  per-feature evidence (PNG frames, distance histograms, tracking error, terrain
  level histograms) into `logs/showcase/<slug>/` so the visual + numerical behaviour
  of every building block can be eyeballed without launching an RL training run.
- `InteractiveScene.viewer_closed` / `ManagerBasedRlEnv.viewer_closed`: viewer-closed
  handling moved into the kernel. `InteractiveScene.step` catches Genesis's
  `GenesisException("Viewer closed.")`, flips the flag, and becomes a no-op on
  subsequent calls. Consumers (RL runner, showcase runner, custom rollouts) only
  need to poll `env.viewer_closed` and break; the previous per-consumer
  `try/except gs.GenesisException` boilerplate is no longer required.
- `InteractiveSceneCfg.batch_render: bool`: when set, `InteractiveScene._build`
  passes `gs.renderers.BatchRenderer(use_rasterizer=False)` to `gs.Scene`. Required
  for `CameraSensor` to produce per-env RGB-D tensors. Defaults to `False` for
  backwards compatibility.
- `Sensor.pre_build_genesis(gs_scene, entities)`: new no-op hook on the sensor base
  class. `InteractiveScene._build` calls it on every sensor cfg right after entity
  spawn and before `gs_scene.build`, giving sensors a chance to register Genesis
  resources that the renderer snapshots at build time. `CameraSensor` overrides it
  to allocate its camera early — required for BatchRender to register the camera in
  its build-time snapshot. `InteractiveScene.sensors` now exposes the resulting
  instances, and `ManagerBasedRlEnv` binds those instances instead of re-building.
- `docs/concepts/scene.{en,zh}.md`: bilingual concept page covering the M1 stack —
  `InteractiveScene` lifecycle, `SimulationCfg`, `InteractiveSceneCfg`, `Articulation`,
  `RigidObject`, and the four documented failure modes.
- `docs/examples/showcase.{en,zh}.md`, `docs/examples/unitree-g1.{en,zh}.md`:
  bilingual examples pages — the showcase index plus a distilled Unitree G1
  walk-through (velocity tracking, motion imitation, long-run profiling).
- `CameraSensor` (with `CameraSensorCfg` and `CameraData`): rigid-mount RGB-D probe
  wrapping Genesis's `BatchRenderer` camera. Per-env tensors with shape
  `(num_envs, H, W, 3)` for RGB (uint8) and `(num_envs, H, W)` for depth (float meters);
  both channels independently toggleable via `render_rgb` / `render_depth`. `bind()`
  resolves the named link, builds the 4×4 mounting offset from `offset_pos` /
  `offset_quat`, and calls `cam.attach`; `_compute_data()` runs `cam.move_to_attach()`
  then `cam.render(...)`. Linux x86-64 + CUDA only — Genesis's `BatchRenderer` is the
  only parallel-env camera backend.
- `genelab.asset_zoo` now ships five built-in robots: `CartpoleCfg`, `FrankaPandaCfg`,
  `UnitreeG1Cfg` (29-DoF humanoid, 6 DCMotor groups mirroring
  `examples/unitree/.../g1/constants.py`), `UnitreeGo1Cfg` (12-DoF quadruped, 3
  ImplicitPD groups aligned to Isaac Lab Go1 defaults), and `AnymalCCfg` (12-DoF
  quadruped, single ImplicitPD group aligned to Isaac Lab Anymal C defaults). All four
  Menagerie-sourced robots (Franka, G1, Go1, Anymal C) ship as `.tar.gz` bundles in
  `KraHsu/genelab-assets`, preserving the upstream LICENSE and README.
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

### Changed

- **Breaking** (asset zoo only — no released API): `FrankaPandaCfg` now sources the full
  MuJoCo Menagerie `franka_emika_panda` model (Apache-2.0) instead of the minimal stub
  shipped during initial M4. Joint regexes drop the `panda_` prefix (`joint[1-7]`,
  `finger_joint.*`) to match the upstream MJCF, and `default_joint_pos` follows the
  Menagerie `home` keyframe (`joint4=-1.57079`, `joint6=1.57079`, `joint7=-0.7853`).
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
