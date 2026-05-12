# Wuji Hand Example

The Wuji hand example registers `GeneLab-Wuji-Hand-Playback-v0`, a fixed-trajectory Genesis scene.
It loads the Wuji MJCF model from bundled description assets and loops the bundled pre-recorded
trajectory using Genesis position control.

Commands below load the example extension directly from this checkout. If you already installed
`examples/genelab_examples`, you can omit the `PYTHONPATH=... --import genelab_examples.tasks`
prefix.

## Run

Run a short headless playback:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 5
```

Run with the Genesis viewer:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --vis --env.robot.side right
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --vis --env.robot.side left
```

## Config Overrides

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --env.reset_interval 0
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --env.robot.side left
```

Wuji-specific config groups:

- `env.robot.side`: `left` or `right`; default `right`.
- `env.robot.desc_dir`: Wuji description directory; defaults to the bundled package assets.
- `env.robot.trajectory`: playback trajectory; defaults to the bundled `wave.npy`.
- `env.reset_interval`: hard-reset interval in steps; default `500`, and `0` disables periodic
  reset.

## Compatibility Wrapper

The wrapper keeps the old fixed-trajectory script shape:

```bash
PYTHONPATH=examples/genelab_examples/src uv run python examples/wuji_hand/fixed_trajectory.py --steps 5
PYTHONPATH=examples/genelab_examples/src uv run python examples/wuji_hand/fixed_trajectory.py --vis --side right
```
