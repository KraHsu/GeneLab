# Extensions

GeneLab core ships **no** robots, environments, or tasks. All content lives in downstream
extension packages. The CLI discovers extensions through three pathways, in order of
preference.

## 1. Entry-point auto-discovery (recommended)

Declare in the extension's `pyproject.toml`:

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project:register"
```

The CLI auto-imports every entry point in the `genelab.extensions` group on startup.
`register()` is a no-argument callable at the package top level that performs the actual
`ROBOTS.register(...)` / `ENVS.register(...)` / `TASKS.register(...)` calls (or simply imports
modules that perform them as a side effect).

This is the pathway `genelab project new` wires up automatically.

## 2. Explicit `--import`

```bash
uv run genelab --import my_pkg.module1 --import my_pkg.module2 list tasks
```

Repeatable. Common uses:

- The extension does not (yet) ship a `genelab.extensions` entry point.
- A fully explicit, reproducible loading order is required (combine with `--no-entry-points`).
- Iterating on a not-yet-installed package while ad-hoc-adding its source directory to
  `sys.path` (the CLI also adds the current working directory to `sys.path`).

## 3. Programmatic

```python
from genelab.registry import load_extension_module

load_extension_module("my_pkg.module")
```

Used by tests and embedding scripts. The same idempotency guard applies, so calling it after
the CLI has already auto-discovered the same module is a no-op.

## Disabling auto-discovery

```bash
uv run genelab --no-entry-points --import my_pkg list tasks
```

This is the most reproducible setup — only the modules explicitly named are loaded.

## Reference extension

`examples/genelab_examples/` is the reference shape: a `pyproject.toml` with the entry point, a
top-level `register()` callable, and `config.py` / `robots.py` / `envs.py` / `tasks.py`
modules. `tests/fake_extension.py` is the same shape stripped to bare minimum for the test
suite.

## See also

- [Project new](../cli/project-new.md)
- [Registry](registry.md)
