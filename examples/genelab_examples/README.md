# GeneLab Example Extension

This directory is a normal external GeneLab project that registers two tasks:

- `GeneLab-Rubiks-Play-v0`
- `GeneLab-GUI-Panels-Demo-v0` — cookbook showing how to add common ImGui viewer widgets
  (sliders, checkboxes, dropdowns, color pickers, …) to the Genesis viewer via
  `SimulationCfg.panels`. One copy-paste recipe per widget family lives in
  `src/genelab_examples/gui_panels/widgets.py`. Needs the ImGui overlay dependency:
  `uv sync --extra imgui`.

From the repository root, run the example without installing this package by importing its
registration module:

```bash
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks list tasks
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 240
PYTHONPATH=examples/genelab_examples/src uv run genelab --import genelab_examples.tasks play GeneLab-GUI-Panels-Demo-v0 --vis
```

Or install the example project once, then use the entry point auto-loading path:

```bash
uv pip install -e examples/genelab_examples
uv run genelab list tasks
uv run genelab play GeneLab-Rubiks-Play-v0 --steps 240
```

Use `--no-entry-points` when you want to verify core GeneLab without installed extensions.
