# Examples

The repository ships several reference extensions under `examples/`. They double as integration
tests for the CLI and registry.

## inverted_pendulum

Two PPO cart-pole tasks built on the same `ManagerBasedRlEnv` + rsl_rl stack as the Unitree
example, sized to fit in a laptop training budget:

- **`GeneLab-Inverted-Pendulum-v0`** — single inverted pole on a cart.
- **`GeneLab-Double-Inverted-Pendulum-v0`** — two stacked inverted poles on a cart.

Source at `examples/inverted_pendulum/`; walkthrough at [Inverted Pendulum](inverted-pendulum.md).

## genelab_examples

The canonical in-tree extension, wiring two tasks:

- **`wuji_hand`** — a hand-manipulation task.
- **`rubiks`** — a Rubik's cube task.

`pyproject.toml` declares the `genelab.extensions` entry point, so the extension is discovered
automatically once installed. The project's `pyproject.toml` also adds the source directory to
pytest's `pythonpath`, so tests can import from it without installation. Source at
`examples/genelab_examples/`.

## unitree

Two PPO tasks on the Unitree G1 humanoid — velocity tracking and motion imitation — ported from
mjlab and adapted to Genesis. Same extension shape as `genelab_examples` (entry point,
`register()`, per-module registration files). Source at `examples/unitree/`.

A complete hands-on walkthrough (install, train, checkpoint replay, motion imitation) lives in
[Quickstart §5](../getting-started/quickstart.md#unitree-g1).

## external_project

A minimal downstream project template. `genelab project new` produces a project of the same
shape; this directory is kept in-tree as a reference for the scaffolding output. Source at
`examples/external_project/`.

## See also

- [Quickstart](../getting-started/quickstart.md)
- [Extensions](../concepts/extensions.md)
