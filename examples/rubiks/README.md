# Rubik's Cube Example

The Rubik's cube example registers `GeneLab-Rubiks-Play-v0`, a force-driven Genesis scene. The cube
is generated as a 27-cubie articulated rigid body with physical collision boxes and visual sticker
geometry.

Commands below load the example extension directly from this checkout. If you already installed
`examples/genelab_examples`, you can omit the `PYTHONPATH=... --import genelab_examples.tasks`
prefix.

## Run

Run a short headless smoke simulation:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5
```

Run a longer simulation:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 240
```

Interactive force mode requires the Genesis viewer:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --vis --steps 10000 --env.interaction.interactive_force true
```

Controls:

- Left-click and drag a cubie to apply spring forces at the picked point.
- Scroll while dragging to rotate the drag plane.
- Close the viewer window or interrupt the process to stop.
- The default mouse spring is intentionally soft and capped to avoid launching light cubies.

## Config Overrides

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5 --env.robot.cubie_size 0.04 --env.robot.gap 0.002
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --env.robot.welded true
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --env.force_controller.enter_ang_vel 0.5
```

Rubik-specific config groups:

- `env.robot.*`: cubie size, gap, welded mode, optional `asset_output`, spawn/material settings.
- `env.interaction.*`: mouse interaction settings, including `interactive_force` and force caps.
- `env.force_controller.*`: thresholds and gains for force-driven mode switching.
- `env.legacy_turn.*`: optional compatibility path for animation-driven layer turns.

Pass `--env.robot.welded true` only when you want one static solid object that cannot enter
three-link mode.

## Controller Behavior

`GeneLab-Rubiks-Play-v0` uses `ForceDrivenCubeController` by default. In simulation, the controller
projects all cubies onto a single rigid lattice. When external force creates enough angular velocity,
the controller chooses the dominant `x`, `y`, or `z` axis and switches into a physical three-link
mode: negative slab, center slab, and positive slab. Each slab is projected as one rigid body, and
side slabs can only rotate around the selected axis relative to the center slab.

## Torque Probe

A lower-level torque probe is kept as an example script:

```bash
PYTHONPATH=examples/genelab_examples/src uv run python examples/rubiks/torque_probe.py --steps 120
```

## Troubleshooting

- If the cube never enters three-link mode, lower `--env.force_controller.enter_ang_vel` or apply a
  stronger external force.
- If a layer is hard to turn, increase `--env.force_controller.turn_gain` or lower
  `--env.force_controller.joint_spring`.
- If the cube exits three-link mode too quickly, lower `--env.force_controller.exit_angle` or
  increase `--env.force_controller.settle_steps`.
