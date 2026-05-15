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
| `prof open` | Launch TensorBoard against a `torch.profiler` trace directory. |
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

## Shell completion

`--install-completion` writes a completion script into the current shell's rc file
(bash, zsh, fish, PowerShell); `--show-completion` prints it to stdout for manual
installation. After installation, tab-completing `genelab info <TAB>` cycles through
every registered task, env, and robot name; `genelab play <TAB>` and
`genelab train <TAB>` cycle through task names; `genelab list <TAB>` offers
`robots / envs / tasks`.

!!! note "Entry-point extensions only"

    Completion callbacks load extensions through the `genelab.extensions`
    entry-point group. Ad-hoc `--import MODULE` registrations do not appear
    in the completion list — the shell strips global flags from the argv it
    hands the callback, so the imported module list is invisible at that
    point.

## Interactive recovery

When stdin is a TTY, four user-input mistakes fall back to a `questionary`
picker instead of exiting with a one-line error:

- `play` / `train` invoked with no task id, or with one that does not match
  any registered task.
- `info NAME` with an unknown name.
- `--agent KIND` with a value other than `zero` / `random` / `trained`.
- `--<a.b.c>` override paths that do not exist on the resolved task's cfg.

Outside a TTY — CI, pipes, scripts, pytest — the picker no-ops and the
original error surfaces unchanged, so non-interactive callers observe
identical behavior to a build without the recovery layer.

## See also

- [Play and Train](play-train.md)
- [Project new](project-new.md)
- [Extensions](../concepts/extensions.md)
