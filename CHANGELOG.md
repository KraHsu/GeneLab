# Changelog

All notable changes to GeneLab are recorded here.

## [Unreleased]

### Added

- **G1 velocity tracking on rough terrain:** added `Genelab-Velocity-Rough-Unitree-G1-v0`,
  a 10-level mixed-terrain curriculum (stairs / boxes / random rough / slopes, mirroring
  Isaac Lab's `ROUGH_TERRAINS_CFG`) driven by `terrain_levels_vel` with capability-relative
  demotion, a `height_scan` actor observation, and a 6k-iter PPO budget. Reference: 4-seed
  sweep, deterministic eval `return_mean` 84–87 on rough terrain (`docs/best-practices/reference-runs`).
- **In-viewport ImGui panels:** `SimulationCfg.panels` takes a list of
  `callback(imgui)` functions that GeneLab forwards to Genesis's ImGui overlay
  after build (a non-empty list auto-enables the overlay). Adding a viewer GUI
  element is now one function — no `Bridge` subclass required. New public helpers
  `register_viewer_panels` / `find_imgui_panel_host` exported from `genelab.scene`;
  new `imgui` extra (`imgui-bundle`). See the `GeneLab-GUI-Panels-Demo-v0` cookbook
  in `examples/genelab_examples`.
- **ImGui twist bridge seeding:** `ImGuiTwistBridgeCfg` gained `default_vx/vy/wz`,
  seeded into the slider state and command buffer at attach (parity with the
  removed DearPyGui bridge).

### Fixed

- **G1 rough-terrain PPO de-learning:** the rough velocity policy would reach a good walk
  and then unlearn it — reward decaying back toward noise as training continued and the
  action std inflating, even on a *fixed* terrain distribution. Root cause: the PPO
  **entropy bonus**. Near convergence the advantages vanish, so the surrogate gradient
  dies and the entropy bonus — however small — dominates and inflates the action std until
  the policy diffuses back to noise (and the adaptive LR, floored by the resulting KL
  spikes, can't recover). Fixed by setting `entropy_coef = 0` and replacing the
  exploration it supplied with a **per-state (heteroscedastic) action std** carrying an
  exploration floor (`std_range = (0.3, 2.0)`): the policy keeps a minimum action noise
  without an unbounded bonus pushing it up, and now trains stably to convergence while the
  terrain curriculum climbs. Two supporting fixes were also required: `feet_swing_height`
  measured peak foot height in absolute world z with an unbounded squared error (spiking
  the cost and diverging the value function on rough/elevated terrain and curriculum
  reseats) — it now measures height relative to the terrain surface under each foot and
  clamps the error; and the rough task lightens the heavy `foot_clearance` /
  `foot_swing_height` shaping penalties (`-2.0 → -0.5`, `-1.0 → -0.25`) so they don't
  overwhelm velocity tracking as the ground roughens (the flat task keeps the heavier
  weights). `terrain_levels_vel` also gains Isaac-Lab-aligned capability-relative demotion.

- **Terrain level curriculum collapsing to a single difficulty:** `TerrainGenerator`
  passed `subterrain_parameters` to Genesis keyed by terrain *type*, so a layout that
  reused one `genesis_type` at different parameters (e.g. a RandomRough `rough_l0..l4`
  level curriculum) silently built every cell at the last level's difficulty — the
  intended easy→hard gradient did not exist. `TerrainGenerator.build_height_field` now
  generates each cell from its own parameters and the importer feeds it to Genesis via
  `height_field`, so per-cell difficulty is honoured. Affects every same-type
  multi-difficulty layout, including `Genelab-Velocity-Rough-Unitree-G1-v0` and the
  curriculum showcase.
- **ImGui overlay close crash:** closing a viewer built with `viewer_imgui` /
  `panels` no longer exits with `GenesisException: Unexpected viewer error.` —
  `InteractiveScene` now wraps Genesis's `ImGuiOverlayPlugin.on_close` (a
  Genesis/`imgui_bundle` teardown bug that asserts `No current context` on window
  close), mirroring the existing pyrender save-filename patch.

### Removed

- **DearPyGui twist bridge and `teleop` extra:** `genelab.bridges.dearpygui` and
  the `teleop` (DearPyGui) extra are removed — the in-viewport `ImGuiTwistBridge`
  covers the same `(vx, vy, ωz)` teleop with no separate window, thread, or third
  GUI toolkit. Migrate `--extra teleop` → `--extra imgui` and
  `DearPyGuiTwistBridgeCfg` → `ImGuiTwistBridgeCfg` (plus `simulation.viewer_imgui=True`).

### Examples

- **ImGui migration:** the Unitree G1 play teleop and the `actuators` showcase
  sliders now render in the viewport ImGui overlay instead of a separate DearPyGui /
  PyQt window (the actuator tracking curves stay in their pyqtgraph window).

## [0.3.1] — 2026-05-31

Patch release. Adds a Genesis-backed mesh ray-caster to the sensor zoo and a
live-image recording sink, and converts every bundled `genelab_showcase`
example to real-time on-screen display instead of writing files to disk.

### Added

- **Mesh ray-cast sensor:** added `MeshRayCastSensor` (with `MeshRayCastSensorCfg`
  and `MeshRayCastData`) over Genesis's native BVH `Raycaster`, plus the
  `MeshGridPattern` / `MeshSphericalPattern` ray patterns — exported from
  `genelab.sensor` and `genelab.lab`. It complements the existing analytic
  `RayCastSensor` (terrain height-field scan) by ray-casting real mesh geometry.
  `RigidObjectCfg.use_visual_raycasting` opts an object's visual mesh into the BVH.
- **Live-image recording sink:** added `MPLImagePlotCfg` to `genelab.recording`,
  a Matplotlib image output for camera-style `(H, W, C)` recording sources.

### Fixed

- **Recorder warning noise:** the recording bridge now filters Genesis's repeated
  `start_thread(): Processor thread already exists` warning around `save_on_reset`
  flushes (Genesis restarts the still-running recorder thread each episode; the
  restart is a harmless no-op). Benefits any env with threaded `save_on_reset`
  recorders, not only the showcases.

### Examples

- **Real-time showcases:** converted all eight bundled `genelab_showcase` examples
  (sensors, ray-cast, contact, terrain, curriculum, actuator, MLP-residual actuator,
  recording) from silent disk-dumping to live Qt / pyqtgraph display. Added a shared
  `LazyQtWindows` helper and a virtual-spring base class for the showcase runners;
  file dumps, where still available, are opt-in and documented.

## [0.3.0] — 2026-05-29

Third development release. The headline is the **Genesis 1.0 migration**:
the simulator floor moves to `genesis-world>=1.0.0,<2.0.0` and the v1.0 API
surface is the baseline for sensors, actuators, scene helpers, and robot
loading. Alongside the migration this release lands a CLI command group
for managing the asset zoo and finishes the per-CLI download-progress-bar
coverage.

### Added

- **Architecture contracts:** added `genelab.contracts` as the public home for
  `EnvContext`, `SceneContext`, and `NoiseCfg`, so domain terms can type-hint the env/scene
  surfaces they need without importing concrete runtime implementations.
- **Typed backend dispatch:** added the `BackendConfig` marker base in `genelab.rl.config`.
  RL backends now register against `type[BackendConfig]`, and `select_backend(agent_cfg)`
  dispatches through the typed registry.
- **Public API guardrails:** made `genelab.lab` a lazy facade and added snapshot coverage for
  the blessed `lab.__all__` and `extensions.__all__` public surfaces.
- **MDP package structure:** split rewards into the `mdp/rewards` package while preserving the
  existing `genelab.mdp` import surface.
- **Sensor zoo:** added wrappers over three Genesis 1.0 sensor primitives —
  `ProximitySensor` (over `gs.sensors.SurfaceDistanceProbe`),
  `KinematicContactSensor` (over `gs.sensors.ContactProbe`), and
  `TemperatureGridSensor` (over `gs.sensors.TemperatureGrid`). Each exposes
  `history_length: int = 0` on its Cfg, forwarded verbatim to the Genesis
  sensor constructor so a single integer flips the ring-buffer on at the
  Genesis layer rather than in user code. Bilingual reference pages under
  `docs/reference/sensors/`.
- **Tactile family:** added two more wrappers in the same shape —
  `ElastomerTactileSensor` (over `gs.sensors.ElastomerTaxel`, deformable
  elastomer model with Lamé `lambda_d` / `lambda_s`) and
  `PointCloudTactileSensor` (over `gs.sensors.ProximityTaxel`, lighter
  stiffness-and-shear model). Both sample a point cloud from the parent
  link's mesh. Added `genelab.mdp.rewards.tactile` with two reward
  primitives generic over `sensor.data.raw`: `contact_intensity_l2`
  (`Σ raw²`) and `contact_count` (count above threshold). Bilingual
  reference pages.
- **Tactile slip reward + integration coverage:** added
  `genelab.mdp.rewards.slip_penalty(env, sensor_name)` — squared sum of
  the lateral (xy) components of `data.raw`, valid for both elastomer
  displacement and point-cloud force outputs (both expose `(B, [H,] P, 3)`
  with the third channel as the probe normal). Added a real-Genesis
  integration test that exercises all four new sensor wrappers via
  `gs.Scene + add_sensor + build + step`, validating the wrappers'
  `pre_build_genesis` / `_compute_data` round-trip end-to-end.
- **Energy rewards:** added `genelab.mdp.rewards.{kinetic_energy_l2,
  potential_energy, energy_budget}` wrapping Genesis 1.0's
  `gs_handle.{get_kinetic_energy, get_potential_energy, get_total_energy}`.
  `kinetic_energy_l2` and `energy_budget` are non-negative (squared);
  `potential_energy` is signed.
- **MJCF-style actuator cfg:** added `MujocoStyleActuatorCfg`, a thin
  cfg over `IdealPDActuator` that accepts Mujoco's `(gear, bias_prm,
  dyntype="none", gaintype="fixed", biastype="affine")` shape and
  translates it into `(stiffness, damping, effort_limit)`. Mirrors what
  Genesis's own MJCF parser does for `<actuator>` elements at load
  time, so robots whose actuator parameters live in Python config —
  rather than in an MJCF file — can use the same named knobs without
  hand-translating to PD gains.
- **Camera debug helpers:** added
  `InteractiveScene.draw_camera_frustums(camera_names=None, color=...)`
  and `InteractiveScene.draw_camera_trajectory(positions, radius, color)`
  wrapping Genesis 1.0's `scene.draw_debug_frustum` /
  `scene.draw_debug_trajectory`. The frustum helper resolves names
  against the scene's `CameraSensor` set; the trajectory helper passes
  positions through verbatim. Added a public `gs_camera` property on
  `CameraSensor` so the scene helper can reach the Genesis handle
  without poking private state.
- **URDF / xacro support on `ArticulationCfg`:** added `urdf_path` and
  `xacro_args` fields. `Articulation.spawn` dispatches on which path is
  set — `mjcf_path` → `gs.morphs.MJCF`, `urdf_path` → `gs.morphs.URDF`
  (which auto-preprocesses `.xacro` / `.urdf.xacro` via the `xacro`
  package at load time). Exactly one of the two paths must be set;
  `xacro_args` is rejected when `mjcf_path` is used.
- **ImGui twist bridge:** new `src/genelab/bridges/imgui.py` with
  `ImGuiTwistBridge` and `ImGuiTwistBridgeCfg`. Same `(vx, vy, ωz)`
  output shape as the keyboard / DearPyGui bridges, but renders three
  native `imgui.slider_float` widgets in Genesis's built-in
  `ImGuiOverlayPlugin` (enabled via `SimulationCfg.viewer_imgui=True`
  from S1) so no extra GUI dependency is required. Falls back to a
  warned no-op when the plugin isn't attached (headless / `viewer_imgui`
  off).
- **Bundled `sample-arm` xacro showcase:** added a minimal 2-DoF arm
  under `genelab.asset_zoo` (`SampleArmCfg(link_length=…)`), shipping
  the xacro file inside the wheel at
  `genelab/asset_zoo/data/sample_arm.urdf.xacro`. Demonstrates
  `ArticulationCfg.urdf_path` + `xacro_args` end-to-end: a single
  `link_length` argument flows through `xacro:arg`, the preprocessor
  substitutes `${L}` and derived `${L * 0.6}` sizes, and Genesis's
  `gs.morphs.URDF` consumes the result. Unlike the other asset-zoo
  entries the file ships in the wheel rather than via the
  `KraHsu/genelab-assets` mirror.
- **`Avatar` kinematic entity:** added `src/genelab/entity/avatar.py`
  with `Avatar` + `AvatarCfg`, a sibling to `RigidObject` that wraps
  Genesis 1.0's `KinematicEntity` (the v0.4.1 "avatar" concept rebadged
  in v1.0). Avatars participate in the visual / sensor pipeline but are
  ignored by the rigid solver — no contacts, no constraints — making
  them the right shape for scripted scene actors (motion-clip ghosts,
  target reference markers, follow-the-leader demonstrators). Per-step
  pose updates flow through `avatar.set_pose(pos, quat)`.
- **`InteractiveSceneCfg.use_rasterizer` toggle:** added a new bool
  field (default `False`) that's forwarded to
  `gs.renderers.BatchRenderer(use_rasterizer=…)` when `batch_render=True`.
  `False` keeps the raytracer (default, higher fidelity), `True`
  switches to the rasterizer (faster batched rendering when
  photorealism isn't required). Ignored when `batch_render=False`.
- **`genelab asset` CLI command group:** added `asset {list,info,download,purge}`
  for managing the bundled asset-zoo cache. `asset list` enumerates every
  `AssetSpec` declared under `genelab.asset_zoo` with downloaded /
  not-downloaded status + on-disk size; `asset info NAME` shows URL / md5 /
  cache path; `asset download NAME` (or `--all`, `--force`) explicitly
  downloads through the existing `fetch_asset` / md5-verify path;
  `asset purge NAME` (or `--all`, `--yes`) removes a cached asset.
  Discovery walks the zoo's submodules, so adding a new robot automatically
  shows up under `asset list` without any registration call.

### Changed

- **Bumped genesis-world to v1.0.0** (`>=1.0.0,<2.0.0`). Picks up non-convex multi-contact
  collision detection, contact pruning, several upstream bug fixes (static-contact drift,
  spurious yaw on flat terrain), and the rigid-solver speedup on contact-rich scenes.
  Exposes the new ImGui debug overlay via `SimulationCfg.viewer_imgui` (forwarded to
  `gs.options.ViewerOptions(enable_gui=...)`; off by default).
- Removed three pre-0.4.7 compatibility shims (`envs_idx=` try/except in
  `_articulation_writer`, `PyQtPlot`/`MPLPlot` recorder-name fallback in `recording.register`,
  and the `gs.device` getattr softprobe in `InteractiveScene.build`) now that the v1.0 API
  surface is the floor.
- `KinematicContactSensor.data`: dropped the mis-named `depth` field
  (Genesis `ContactProbe.read()` returns a bool, not a depth) and the
  redundant threshold operation. `data.in_contact` is now the
  `(num_envs, [history,] num_probes)` `bool` tensor Genesis returns,
  already thresholded at `cfg.contact_threshold`.
- `PointCloudTactileSensor.data`: replaced the single `raw` field with
  `force` and `torque` tensors extracted from Genesis's structured
  `ProximityTaxelData`. `raw` is kept as a property alias for `force` so
  the shape-agnostic reward primitives still see the right tensor.
- **Asset-download progress bar reaches every CLI surface.** `genelab export`
  now wraps its task-construction step in the same `fetch_progress()` Rich
  callback every other CLI surface uses (`play`, `train`, `eval`, `info`, the
  new `asset *` subcommands), so a first-touch asset download from any command
  renders a progress bar instead of staying silent. The download infrastructure
  itself was already callback-driven; only `_export` was missing the
  `with fetch_progress():` block.
- **Breaking:** removed `lab.GenesisBackendCfg`.
- **Breaking:** renamed the environment registration protocol from `lab.ManagerBasedEnv` to
  `lab.ManagerBasedEnvProtocol`.
- Moved backend-specific config modules from `genelab.rl.{skrl_config,skrl_models,sb3_config}`
  to `genelab.rl.backends.{skrl,sb3}.{config,models}`. The `genelab.rl.*Cfg` facades remain
  available for user code.
- Moved `NoiseCfg`'s canonical home to `genelab.contracts`; it remains re-exported from the MDP
  noise module and the public facades.

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
