# Examples

Examples are extension packages that exercise GeneLab capabilities without adding project-specific
code to `src/genelab/`.

## Capability map

| Example | Package | Shows |
|---|---|---|
| [Inverted Pendulum](inverted-pendulum.md) | `examples/inverted_pendulum` | Minimal train/play loop, manager terms, RSL-RL integration. |
| [Unitree G1](unitree-g1.md) | `examples/unitree` | Humanoid locomotion, velocity commands, motion imitation. |
| [Franka Pick-and-Place](franka-pick-and-place.md) | `examples/franka_pick_and_place` | Goal-conditioned manipulation, SAC + HER + lift bonus + FSM demo prefill. |
| [Showcase](showcase.md) | `examples/genelab_showcase` | Sensors, ray casts, contact, terrains, curricula, actuators, recording. |
| [Rubik's Cube](rubiks-cube.md) | `examples/genelab_examples` | Rigid-object composition and visual interaction. |
| [Wuji Hand](wuji-hand.md) | `examples/genelab_examples` | Articulated hand playback and asset packaging. |

## Installing examples

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
uv pip install -e examples/franka_pick_and_place
genelab list tasks
```

Install Unitree only when needed:

```bash
uv pip install -e examples/unitree
```

## See also

- [Tutorial](../tutorial.md)
- [Build an Extension Project](../best-practices/extension-projects.md)
