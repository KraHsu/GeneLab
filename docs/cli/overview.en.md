# CLI overview

`genelab` is exposed as a console script via the `genelab = "genelab.cli:main"` entry point.
With `uv` the canonical invocation is:

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

The following flags accept any position in front of the subcommand:

| Flag | Effect |
|------|--------|
| `--version` | Print the GeneLab version and exit. |
| `--import MODULE` | Eagerly import an extension module before dispatching. Repeatable. Useful for extensions that do not (yet) ship a `genelab.extensions` entry point. |
| `--no-entry-points` | Skip auto-discovery via the `genelab.extensions` entry-point group. Combined with `--import`, produces a fully explicit, reproducible loading order. |

## Extension discovery order

On startup the CLI discovers extensions through three pathways, in order: entry-point
auto-discovery, explicit `--import MODULE` flags, then programmatic
`genelab.registry.load_extension_module(...)` calls.

## See also

- [Play and Train](play-train.md)
- [Project new](project-new.md)
- [Extensions](../concepts/extensions.md)
