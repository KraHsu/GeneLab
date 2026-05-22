# Changelog

All notable changes to GeneLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is still on a 0.x
trajectory so breaking changes can land in any minor release until the 1.0 stabilisation.

## [Unreleased]

### Changed

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
