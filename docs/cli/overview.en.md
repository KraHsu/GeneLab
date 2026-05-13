# CLI overview

`genelab` is exposed as a console script via the `genelab = "genelab.cli:main"` entry point. With
`uv` you typically invoke it as:

```bash
uv run genelab [GLOBAL OPTIONS] <subcommand> [ARGS]
```

## Subcommands

| Subcommand | Purpose |
|------------|---------|
| `cache` | Create project-local simulation cache directories (`.cache/`) and set `XDG_CACHE_HOME` / `MPLCONFIGDIR`. |
| `list robots` | List registered robots from the `ROBOTS` registry. |
| `list envs` | List registered environments from the `ENVS` registry. |
| `list tasks` | List registered tasks from the `TASKS` registry. |
| `play` | Run a registered task in the configured Genesis backend; supports config overrides. |
| `train` | Train a registered task when an RL runner exists; supports multi-GPU via `torchrun`. |
| `project new` | Scaffold a new external extension package with all three registries wired up. |

## Global options

These flags work in front of any subcommand:

- `--version` — print the GeneLab version and exit.
- `--import MODULE` — eagerly import an extension module before dispatching. Repeatable. Useful
  when you want to load an extension that does not (yet) ship a `genelab.extensions` entry point.
- `--no-entry-points` — skip auto-discovery of installed extensions via the `genelab.extensions`
  entry-point group. Combine with `--import` for fully explicit, reproducible loading.

## Extension loading

When the CLI starts, it discovers extensions in this order:

1. **Entry points** under the `genelab.extensions` group (auto, unless `--no-entry-points`).
2. **Explicit `--import MODULE` flags** (repeatable).
3. **Programmatic** `genelab.registry.load_extension_module(...)` (used by tests and embedding
   scripts).

See [Extensions](../concepts/extensions.md) for details on writing a downstream extension.

## See also

- [Play and Train](play-train.md) — config override grammar, multi-GPU training, checkpoints.
- [Project new](project-new.md) — extension package scaffolding.
