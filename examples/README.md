# GeneLab Examples

This directory contains runnable examples and downstream extension packages. GeneLab core does not
ship built-in tasks; example tasks are loaded like any other external project.

## Available Examples

- [Rubik's Cube](rubiks/README.md): force-driven Rubik's cube Genesis scene and torque probe script.
- [Wuji Hand](wuji_hand/README.md): fixed-trajectory Wuji hand playback scene and compatibility
  wrapper script.
- [GeneLab Example Extension](genelab_examples/README.md): one Python project that registers the
  Rubik's cube and Wuji hand tasks.
- [External Project](external_project/README.md): minimal standalone Python package that extends
  GeneLab without editing `src/genelab/`.

The example extension registers these task IDs:

- `GeneLab-Rubiks-Play-v0`
- `GeneLab-Wuji-Hand-Playback-v0`

List example tasks from the repository root without installing the example package:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks list tasks
```

Install the example extension once if you want `uv run genelab list tasks` to load it through entry
points:

```bash
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
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5 --env.robot.cubie_size 0.04 --env.robot.gap 0.002
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --env.robot.welded true
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --env.reset_interval 0
```

`train` validates the task id and configuration path but currently reports that training is not
implemented:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks train GeneLab-Rubiks-Play-v0
```

## Smoke Tests

Run short headless smoke tests after the Genesis assets and cache have initialized:

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
