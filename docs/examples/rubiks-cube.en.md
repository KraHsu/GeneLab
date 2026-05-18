# Rubik's Cube

`examples/genelab_examples/src/genelab_examples/rubiks/` ships a play-only Genesis demo of a
force-driven 3×3 Rubik's cube. The task is **not** an RL task — it bypasses
`ManagerBasedRlEnv` and drives a Genesis `Scene` directly through a custom controller. It
exists primarily as an end-to-end example of (1) registering a non-RL task with the GeneLab
registry, (2) generating an MJCF asset on the fly, and (3) using Genesis's dynamic
constraint API to switch a rigid body between articulation modes.

## Task

| Task id | Problem |
|---------|---------|
| `GeneLab-Rubiks-Play-v0` | A 27-cubie cube that dynamically welds into one rigid body, detects when it has enough angular velocity to "twist", and re-constrains itself into three slabs along the rotating axis. |

## Installation

```bash
uv sync --extra torch-cu128
uv pip install -e examples/genelab_examples

uv run genelab list tasks | grep Rubiks
# -> GeneLab-Rubiks-Play-v0
```

Or run without installing by adding the example `src` to `PYTHONPATH`:

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks list tasks
```

## Run

```bash
uv run genelab play GeneLab-Rubiks-Play-v0 --vis --steps 600

# Bigger cubie, smaller gap.
uv run genelab play GeneLab-Rubiks-Play-v0 --steps 5 \
    --env.robot.cubie_size 0.04 --env.robot.gap 0.002

# Statically weld every cubie (mode comparison).
uv run genelab play GeneLab-Rubiks-Play-v0 --env.robot.welded true --steps 5

# Interactive mouse drag (requires --vis).
uv run genelab play GeneLab-Rubiks-Play-v0 --vis --steps 0 \
    --env.interaction.interactive_force true
```

`--steps 0` runs the viewer loop until you close the window. Without `--steps` the env
falls back to the value baked into `SimulationCfg.steps`.

## What this example demonstrates

| GeneLab capability | Where it appears | Concept doc |
|---|---|---|
| Task registration via `register_task` + `TaskCfg` | `tasks.py` | [Registry](../concepts/registry.md) |
| Robot registration via `register_robot` (MJCF written from a `RubiksCubeSpec`) | `robots.py`, `rubiks/assets.py:write_mjcf` | [Registry](../concepts/registry.md), [Asset zoo](../concepts/asset_zoo.md) |
| Configuration dataclass pattern (`RubiksEnvCfg` extends `ManagerBasedEnvCfg`) | `rubiks/config.py` | [Configs](../concepts/configs.md) |
| Custom `play()` runner that bypasses the manager pipeline | `envs.py:RubiksPlayEnv.play` | Contrasts with [Managers and MDP terms](../concepts/managers.md) |
| Genesis dynamic constraint API (weld / hinge / external torque) | `rubiks/sim.py:ForceDrivenCubeController` | [Scene and entities](../concepts/scene.md) |
| Genesis `MouseInteractionPlugin` for live force drags | `envs.py:RubiksPlayEnv.play` | Same plugin as the [inverted pendulum](inverted-pendulum.md#interactive-disturbance) |

## Code walkthrough

The package is split so each file owns one concern:

- **`rubiks/config.py` (49 lines)** — `RubiksEnvCfg`, `RubiksRobotCfg`, `RubiksInteractionCfg`,
  `ForceDrivenCubeConfig`. All overridable through dotted CLI flags
  (`--env.robot.cubie_size 0.04`, `--env.interaction.mouse_spring 25`, ...).
- **`rubiks/assets.py` (249 lines)** — Pure geometry. `RubiksCubeSpec` parameterises cubie
  size, sticker inset, friction, mass, etc. `iter_cubie_coords` and `exposed_faces` enumerate
  the 27-cubie lattice. `write_mjcf` emits the runtime MJCF that Genesis loads.
- **`rubiks/sim.py` (1061 lines)** — The bulk of the example. Two controllers live here:
    - `RubiksCubeController` — legacy kinematic turner (`set_qpos` per layer). Used when
      `legacy_turn.enabled=true` for visual ground truth.
    - `ForceDrivenCubeController` — the default. A state machine that:
        1. Welds all 27 cubies into one rigid body via `solver.add_weld_constraint`.
        2. Watches the welded body's angular velocity. When `|ω| > enter_ang_vel` along an
           axis, it deletes the welds *for that axis only* and adds hinge-constraint pairs
           so the cube splits into three slabs.
        3. Applies a small `joint_spring + joint_damping` torque through
           `solver.apply_links_external_torque` to keep the side slabs at a quantised
           quarter-turn rest position.
        4. When the slab angle settles below `exit_angle` *and* below `exit_ang_vel`, it
           deletes the hinges and re-welds the slabs back together.
    - This length is intrinsic: the cube has 27 cubies × 3 axes × 2 sides × multi-step
      settle logic, and every state transition needs Genesis-constraint bookkeeping. The
      file is not bloated; trimming it would remove behaviour, not noise.
- **`envs.py:RubiksPlayEnv`** — Builds the Genesis `Scene` (rigid options tuned for the
  many-constraint regime: `max_dynamic_constraints=128`, `noslip_iterations=5`), loads the
  generated MJCF, optionally attaches `MouseInteractionPlugin`, and runs the controller
  step-by-step for `cfg.simulation.steps` ticks. **No `ManagerBasedRlEnv` involved** — this
  task does not have observations, actions, or rewards.
- **`tasks.py`** — Registers `GeneLab-Rubiks-Play-v0` as a play-only task
  (`trainable=False`). Invoking `genelab train` on it raises `NotImplementedError`.

## Smoke test

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5
```

A 5-step play run is enough to validate asset generation, scene construction, and one
controller step. The first run compiles Genesis kernels and is slower.

## See also

- [Wuji Hand](wuji-hand.md) — the sibling play-only demo packaged in the same extension.
- [Registry](../concepts/registry.md) — how `register_task` / `register_robot` work.
- [Configs](../concepts/configs.md) — the dataclass override mechanism that
  `--env.robot.welded true` rides on.
- [Scene and entities](../concepts/scene.md) — Genesis primitives used by the controller.
- [Inverted Pendulum](inverted-pendulum.md) — the other example that uses
  `MouseInteractionPlugin`.
