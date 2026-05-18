# Wuji Hand

`examples/genelab_examples/src/genelab_examples/wuji_hand/` ships a play-only Genesis
demo of the Wuji five-finger dexterous hand replaying a fixed `.npy` trajectory. Like the
[Rubik's cube example](rubiks-cube.md) it is **not** an RL task — it builds a Genesis
`Scene` directly and PD-controls each joint toward the next trajectory frame. The point
is to show the minimum amount of GeneLab + Genesis glue needed to drive a robot from a
recorded motion clip.

## Task

| Task id | Problem |
|---------|---------|
| `GeneLab-Wuji-Hand-Playback-v0` | Load the right-side Wuji MJCF, map each `right_fingerN_jointM` to its Genesis DOF, then replay a 20-column trajectory frame-by-frame through `control_dofs_position`. |

## Installation

```bash
uv sync --extra torch-cu128
uv pip install -e examples/genelab_examples

uv run genelab list tasks | grep Wuji
# -> GeneLab-Wuji-Hand-Playback-v0
```

Or run without installing:

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks list tasks
```

## Run

```bash
# Default trajectory: a 20-DoF wave gesture bundled in the package.
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --steps 600

# Disable periodic hard resets (let the trajectory run uninterrupted).
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --env.reset_interval 0

# Swap to the left-side MJCF (only `right` is bundled by default).
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --env.robot.side left
```

`reset_interval` is the number of Genesis steps between periodic teleports back to the
joint-zero pose; `0` disables the teleport. The default `wave.npy` ships next to the
package source.

## What this example demonstrates

| GeneLab capability | Where it appears | Concept doc |
|---|---|---|
| Task registration of a play-only task | `tasks.py` | [Registry](../concepts/registry.md) |
| Robot registration backed by an on-disk MJCF asset directory | `robots.py`, `wuji_hand/assets.py:resolve_mjcf_path` | [Registry](../concepts/registry.md), [Asset zoo](../concepts/asset_zoo.md) |
| Config dataclass with nested `SimulationCfg` and a default `Path` field | `wuji_hand/config.py` | [Configs](../concepts/configs.md) |
| Joint-name → Genesis DOF-index mapping for trajectory replay | `wuji_hand/sim.py:build_joint_mapping`, `JointMapping` dataclass | n/a (example-specific helper) |
| Trajectory loading from `.npy` + per-joint clamping to DOF limits | `wuji_hand/assets.py:load_trajectory`, `wuji_hand/sim.py:trajectory_target` | Same pattern as the [Unitree G1 motion clip](unitree-g1.md#motion-imitation) |
| Per-joint PD gain and force-range setup at runtime | `wuji_hand/sim.py:apply_wuji_gains` | [Actuators](../concepts/actuators.md) — contrast with the configured-actuator pattern |

## Code walkthrough

The package is small (≈400 lines total) and each file owns one concern:

- **`wuji_hand/config.py` (24 lines)** — `WujiEnvCfg` (extends `ManagerBasedEnvCfg`) and
  `WujiRobotCfg`. Default `dt=0.01`, `steps=0` (infinite viewer loop), `reset_interval=500`,
  `side="right"`. All overridable via dotted flags
  (`--env.simulation.dt 0.005`, `--env.reset_interval 0`, `--env.robot.side left`).
- **`wuji_hand/assets.py` (74 lines)** — File-path resolution and trajectory loading.
  - `wuji_joint_names(side)` returns the 20 joint names in the order the trajectory's
    columns expect (`{side}_fingerN_jointM` for `N=1..5, M=1..4`).
  - `resolve_mjcf_path(desc_dir, side)` tries three candidate paths so users can drop in
    their own MJCF layout.
  - `load_trajectory(path)` validates shape (2D, ≥20 columns) and returns `float32`.
- **`wuji_hand/sim.py` (280 lines)** — The playback loop. Worth reading in order:
    1. `WujiHandRunConfig` — flat run-time config with `__post_init__` validators.
    2. `build_joint_mapping(entity, side)` — for each trajectory column, look up the matching
       Genesis joint by name and record its local DOF index. Missing joints are skipped
       with a warning rather than failing.
    3. `apply_wuji_gains` — sets `kp=0.8`, `kv=0.04` on every mapped DOF and copies the
       MJCF's declared force range onto Genesis's runtime range (only when both bounds are
       finite — MJCFs with `forcerange="0 0"` or unset values are passed through).
    4. `run_wuji_hand(config)` — initialises Genesis, builds the scene (with rigid options
       tuned for self-collision on a high-DoF hand), maps joints, then in the main loop
       calls `entity.control_dofs_position(target, mapping.dof_indices)` once per Genesis
       step using the clamped trajectory frame.
- **`envs.py:WujiHandPlaybackEnv.play`** — Thin wrapper that copies `WujiEnvCfg` fields
  into a `WujiHandRunConfig` and calls `run_wuji_hand`. **No `ManagerBasedRlEnv`** — like
  the Rubik's example, this task has no obs/action/reward graph.
- **`tasks.py`** — Registers `GeneLab-Wuji-Hand-Playback-v0` with `trainable=False`.

## Smoke test

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 5
```

A 5-step run validates MJCF resolution, joint mapping, and one trajectory frame. First
launch triggers Genesis kernel compilation.

## See also

- [Rubik's Cube](rubiks-cube.md) — sibling play-only demo packaged in the same extension.
- [Unitree G1](unitree-g1.md) — the larger motion-replay example (LAFAN1 NPZ, full
  humanoid).
- [Registry](../concepts/registry.md) — how the task and robot are wired in.
- [Configs](../concepts/configs.md) — dataclass override mechanism behind the
  `--env.robot.side left` flag.
