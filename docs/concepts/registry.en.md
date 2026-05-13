# Registry

GeneLab provides a generic `Registry[T]` plus three module-level singletons exported from
`genelab.lab`:

| Singleton | Holds |
|-----------|-------|
| `ROBOTS` | Robot factories. |
| `ENVS` | Environment factories. |
| `TASKS` | Task factories (each task pairs an environment with optional runner / agent config). |

## What an entry looks like

A registry entry is a 4-tuple: **`(name, description, factory, cfg_type)`**. The `factory` is a
callable invoked lazily on `get(name)`, so importing a registration site does not eagerly build
heavy objects.

## Basic usage

```python
from genelab.lab import ROBOTS, ENVS, TASKS, TaskCfg

def make_my_robot():
    ...

ROBOTS.register(
    name="my_robot",
    description="A demo robot.",
    factory=make_my_robot,
    cfg_type=None,
)

# Later:
robot = ROBOTS.get("my_robot")
```

The `name` is the canonical lookup key; the `description` shows up in `genelab list robots`.

## Idempotent extension loading

The registry module also tracks `_loaded_extension_modules` and `_loaded_entrypoints` sets so that
repeated calls to load the same extension are no-ops. This matters when a user combines the entry
point auto-discovery (default) with explicit `--import` flags — both pathways end up calling
`load_extension_module`, but the second call is short-circuited.

See [Extensions](extensions.md) for the three pathways and when to use which.

## API

The full auto-generated reference for `genelab.registry` lives on the
[API Reference](../api/reference.md) page.
