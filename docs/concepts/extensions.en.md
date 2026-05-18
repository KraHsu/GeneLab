# Extensions

Extensions are ordinary Python packages that register robots, environments, and tasks with GeneLab.
They are the expected way to build real projects.

## Why extensions are separate

Keeping downstream projects outside `src/genelab/` avoids turning the framework into a collection of
project-specific code. It also lets teams version, install, and publish their robot packages
independently.

## Discovery mechanisms

| Mechanism | Best use |
|---|---|
| Entry point in `genelab.extensions` | Installed packages and daily workflows. |
| CLI `--import MODULE` | Temporary local modules or debugging entry-point loading. |
| Programmatic loading | Embedded applications. |

All mechanisms end with the same operation: a registration function calls `register_robot`,
`register_env`, and `register_task`.

## Extension contract

An extension should be importable as a package, keep registry-time imports light, expose a
no-argument `register()` hook, and avoid duplicate registration in repeated loads.

## Where to continue

- [Build an Extension Project](../best-practices/extension-projects.md)
- [Project New](../cli/project-new.md)
- [Registry](registry.md)
