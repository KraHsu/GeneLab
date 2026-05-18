# Registry

GeneLab uses small registries to keep the framework separate from downstream robot projects.
The core package does not need to know every robot, environment, or task at import time; extensions
register named factories when the CLI or user code asks for them.

## Why registries exist

Robotics projects often mix three concerns: asset definitions, environment construction, and
experiment execution. GeneLab separates them:

| Registry | Boundary |
|---|---|
| `ROBOTS` | Named robot/asset config factories. |
| `ENVS` | Named environment factories. |
| `TASKS` | Named task factories that carry `TaskCfg` and runtime methods. |

This gives the CLI a stable discovery model without forcing a central list of all downstream
projects into `genelab` itself.

## Lazy factories

Registry entries store factories rather than constructed objects. Listing names and descriptions is
cheap; expensive work such as importing Genesis, downloading assets, or building a scene happens
only when the factory is called.

This is why a command like `genelab list tasks` can be fast while `genelab play TASK` is allowed to
initialize the simulator.

## Extension loading

Extensions can register in three ways:

| Mechanism | Use case |
|---|---|
| `genelab.extensions` entry point | Daily use after installing a package. |
| `--import MODULE` | Local experiments, notebooks, and packages not installed as entry points. |
| `load_extension_module()` | Programmatic embedding. |

Registration should be idempotent when a module may be loaded repeatedly in one process.

## Where to continue

- [Build an Extension Project](../best-practices/extension-projects.md)
- [Discovery: list and info](../cli/list-info.md)
- [API Reference](../api/reference.md)
