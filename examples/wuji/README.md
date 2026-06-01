# Wuji Hand Example

This directory is a normal external GeneLab project for the Wuji dexterous hand. It registers:

- `GeneLab-Wuji-Hand-Playback-v0` — fixed-trajectory playback of the Wuji hand.

From the repository root, run the example without installing this package by importing its
registration module:

```bash
PYTHONPATH=examples/wuji/src uv run genelab --import genelab_wuji.tasks list tasks
PYTHONPATH=examples/wuji/src uv run genelab --import genelab_wuji.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 240
```

Or install the example project once, then use the entry-point auto-loading path:

```bash
uv pip install -e examples/wuji
uv run genelab list tasks
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --steps 240
```

Use `--no-entry-points` when you want to verify core GeneLab without installed extensions.
