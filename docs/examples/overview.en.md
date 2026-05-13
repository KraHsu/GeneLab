# Examples

The repository ships several reference extensions under `examples/`. They double as integration
tests for the CLI and registry.

## genelab_examples

Path: [`examples/genelab_examples/`](https://github.com/KraHsu/GeneLab/tree/main/examples/genelab_examples)

The canonical in-tree extension. Two tasks are wired up:

- **`wuji_hand`** — a hand-manipulation task.
- **`rubiks`** — a Rubik's cube task.

`pyproject.toml` declares the `genelab.extensions` entry point, so this extension is discovered
automatically when the package is installed (it is also on `pytest`'s `pythonpath` via the
project's `pyproject.toml`, allowing the tests to import from it without installation).

## unitree

Path: [`examples/unitree/`](https://github.com/KraHsu/GeneLab/tree/main/examples/unitree)

A robot example focused on Unitree platforms. Same extension shape as `genelab_examples` —
entry point, `register()`, and per-module registration files.

## external_project

Path: [`examples/external_project/`](https://github.com/KraHsu/GeneLab/tree/main/examples/external_project)

A minimal downstream project template. This is what `genelab project new` produces, kept in-tree
as a reference for the scaffolding output.

## Using an example

```bash
# Confirm the example tasks are visible.
uv run genelab list tasks

# Play a task with visualization.
uv run genelab play wuji_hand --vis --steps 200
```

## See also

- [Project new](../cli/project-new.md) — scaffold your own extension.
- [Extensions](../concepts/extensions.md) — how extensions are discovered and loaded.
