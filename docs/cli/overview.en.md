# CLI Overview

The `genelab` CLI is a thin dispatcher over registries and task configs. It does not own task logic;
it discovers extensions, resolves a registered object, applies overrides, and calls the task or
runner.

## Command model

```bash
genelab [global options] <command> [arguments]
```

| Area | Commands |
|---|---|
| Registry discovery | `list robots`, `list envs`, `list tasks`, `info NAME` |
| Runtime | `play TASK`, `train TASK` |
| Utilities | `cache`, `prof open` |
| Project scaffolding | `project new NAME` |

## Extension loading order

Every command that needs registry data loads extensions in this order:

1. Bundled asset zoo robots through `load_bundled_asset_zoo()`.
2. Installed `genelab.extensions` entry points, unless `--no-entry-points` is set.
3. Repeated explicit `--import MODULE` values.

Use entry points for daily work and `--import` for local experiments.

## Overrides

Runtime commands accept unknown `--a.b.c VALUE` options after the task id. The CLI forwards them to
`apply_overrides`.

```bash
genelab play TASK_ID --env.simulation.dt 0.005
```

Use `genelab info TASK_ID` to list valid paths.

## Interactive mode

When stdin is a TTY, the CLI can prompt for a missing task id, unknown registry name, invalid
`--agent`, or unknown override path. In CI and scripts it raises the same errors directly.

## See also

- [CLI Reference](../reference/cli.md)
- [Discovery: list and info](list-info.md)
- [Play and Train](play-train.md)
