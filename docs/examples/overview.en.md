# Examples

The repository ships several reference extensions under `examples/`. They double as
integration tests for the CLI and registry. Every runnable example has its own page
under this section; this overview is the entry point.

## Capability map

| Example | Demonstrates | Doc |
|---|---|---|
| `inverted_pendulum` | `ManagerBasedRlEnv` + PPO, `BodyVelocitySensor`, `RecordingCfg`, `MouseInteractionPlugin` | [Inverted Pendulum](inverted-pendulum.md) |
| `unitree` | PPO velocity-tracking + motion imitation on the G1 humanoid, asset zoo + motion-clip pipeline | [Unitree G1](unitree-g1.md) |
| `genelab_showcase` | Seven per-feature minimal play tasks (sensors, ray-cast, contact, terrain, curriculum, actuators, recording) | [Showcase](showcase.md) |
| `genelab_examples/rubiks` | Custom play runner, MJCF generated at runtime, Genesis dynamic constraint API | [Rubik's Cube](rubiks-cube.md) |
| `genelab_examples/wuji_hand` | MJCF asset directory, joint-name → DOF mapping, fixed-trajectory replay | [Wuji Hand](wuji-hand.md) |
| `external_project` | Scaffolding template — same shape as `genelab project new` output | see [Project New CLI](../cli/project-new.md) |

## inverted_pendulum

Two PPO cart-pole tasks built on the same `ManagerBasedRlEnv` + rsl_rl stack as the
Unitree example, sized to fit in a laptop training budget:

- **`GeneLab-Inverted-Pendulum-v0`** — single inverted pole on a cart.
- **`GeneLab-Double-Inverted-Pendulum-v0`** — two stacked inverted poles on a cart.

Source at `examples/inverted_pendulum/`. Full walkthrough on the
[Inverted Pendulum](inverted-pendulum.md) page.

## unitree

Two PPO tasks on the Unitree G1 humanoid — velocity tracking and motion imitation —
ported from mjlab and adapted to Genesis. Same extension shape as `genelab_examples`
(entry point, `register()`, per-module registration files). Source at
`examples/unitree/`. Full walkthrough on the [Unitree G1](unitree-g1.md) page.

## genelab_showcase

Seven play-only tasks, one per GeneLab building block, each driving a real Franka or G1
into a minimal `ManagerBasedRlEnv` and dumping a per-feature evidence file under
`logs/showcase/<slug>/`. Designed for human visual / numerical verification rather than
training. Source at `examples/genelab_showcase/`. Full walkthrough on the
[Showcase](showcase.md) page.

## genelab_examples

The canonical in-tree extension, wiring two **play-only** Genesis demos that bypass
`ManagerBasedRlEnv`:

- **`GeneLab-Rubiks-Play-v0`** — 27-cubie cube that dynamically welds and re-articulates
  itself through Genesis's constraint API. See [Rubik's Cube](rubiks-cube.md).
- **`GeneLab-Wuji-Hand-Playback-v0`** — Wuji five-finger hand replaying a 20-DoF
  `.npy` trajectory through `control_dofs_position`. See [Wuji Hand](wuji-hand.md).

`pyproject.toml` declares the `genelab.extensions` entry point, so the extension is
discovered automatically once installed. The project's `pyproject.toml` also adds the
source directory to pytest's `pythonpath`, so tests can import from it without
installation. Source at `examples/genelab_examples/`.

## external_project

A minimal downstream-project template. `genelab project new` produces a project of the
same shape; this directory is kept in-tree as a reference for the scaffolding output.
It is **not** a runnable example task — instead see the
[Project New CLI](../cli/project-new.md) and [Extensions](../concepts/extensions.md)
pages for how to bootstrap your own downstream project from it. Source at
`examples/external_project/`.

## See also

- [Quickstart](../getting-started/quickstart.md)
- [Extensions](../concepts/extensions.md)
- [Project New CLI](../cli/project-new.md)
