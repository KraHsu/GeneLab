# GeneLab Examples

This directory contains runnable examples and downstream extension packages. GeneLab core does not
ship built-in tasks; example tasks are loaded like any other external project.

## Available Examples

- [Inverted Pendulum](inverted_pendulum/README.md): trainable single- and double-inverted-pendulum tasks that initialize Genesis while fully exercising `train` + `play`.
- [GeneLab Example Extension](genelab_examples/README.md): one Python project that registers the
  Rubik's cube and Wuji hand tasks.
- [External Project](external_project/README.md): minimal standalone Python package that extends
  GeneLab without editing `src/genelab/`.

The bundled examples register these task IDs:

- `GeneLab-Inverted-Pendulum-v0`
- `GeneLab-Double-Inverted-Pendulum-v0`
- `GeneLab-Rubiks-Play-v0`
- `GeneLab-Wuji-Hand-Playback-v0`

List inverted-pendulum tasks from the repository root without installing the package:

```bash
PYTHONPATH=examples/inverted_pendulum/src uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

List the Genesis demo tasks the same way:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks list tasks
```

Install an example extension once if you want `uv run genelab list tasks` to load it through entry
points:

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
uv run genelab list tasks
```

## Common Runner Flags

Common runner flags are accepted after the task id, alongside dotted config overrides:

- `--vis`: show the Genesis viewer.
- `--gpu`: use the Genesis GPU backend instead of CPU.
- `--steps`: number of simulation steps to run.

Any `--path value` pair after the task id is applied to the registered task config. Hyphens in
override keys are converted to underscores.

Examples:

```bash
PYTHONPATH=examples/inverted_pendulum/src uv run genelab --import genelab_inverted_pendulum.tasks train GeneLab-Inverted-Pendulum-v0 --num-envs 4096 --max-iterations 150
PYTHONPATH=examples/inverted_pendulum/src uv run genelab --import genelab_inverted_pendulum.tasks play GeneLab-Inverted-Pendulum-v0 --checkpoint logs/rsl_rl/inverted_pendulum_flat/<run>/model_150.pt --vis
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5 --env.robot.cubie_size 0.04 --env.robot.gap 0.002
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --env.robot.welded true
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --env.reset_interval 0
```

`train` is implemented for the inverted-pendulum tasks. The Rubik's cube and Wuji hand demo tasks are
play-only and report that training is not implemented:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks train GeneLab-Rubiks-Play-v0
```

## Smoke Tests

Run the inverted-pendulum smoke tests first; a tiny rsl_rl run exercises the full Genesis +
PPO pipeline:

```bash
PYTHONPATH=examples/inverted_pendulum/src uv run genelab --import genelab_inverted_pendulum.tasks train GeneLab-Inverted-Pendulum-v0 --num-envs 64 --max-iterations 5
PYTHONPATH=examples/inverted_pendulum/src uv run genelab --import genelab_inverted_pendulum.tasks train GeneLab-Double-Inverted-Pendulum-v0 --num-envs 64 --max-iterations 5
```

Then run short headless smoke tests after the Genesis assets and cache have initialized:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 5
```

The first Genesis run may spend extra time compiling simulation kernels. Add `--vis` only when you
want the Genesis viewer window.

## Troubleshooting

- If the viewer cannot open on Linux, check the available OpenGL platform and GPU/driver setup.
- If an external project cannot be imported, confirm the package is installed or its `src/`
  directory is on `PYTHONPATH`.
- If a task id is unknown, rerun `uv run genelab list tasks` or the `PYTHONPATH=... --import ...`
  command above and confirm the corresponding extension registration hook has loaded.
