# Registry

GeneLab provides a generic `Registry[T]` plus three module-level singletons exported from
`genelab.lab`:

| Singleton | Holds |
|-----------|-------|
| `ROBOTS` | Robot factories. |
| `ENVS` | Environment factories. |
| `TASKS` | Task factories (each task pairs an environment with optional runner / agent config). |

## Entry shape

A registry entry is a 4-tuple: **`(name, description, factory, cfg_type)`**. The `factory` is a
callable invoked lazily on `get(name)`, so importing a registration site does not eagerly build
heavy objects.

## API surface

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

The `name` is the canonical lookup key; the `description` is what `genelab list robots`
displays.

## Idempotent extension loading

The registry module tracks `_loaded_extension_modules` and `_loaded_entrypoints` sets so that
repeated calls to load the same extension are no-ops. This matters when entry-point
auto-discovery (default) combines with explicit `--import` flags — both pathways end up calling
`load_extension_module`, but the second call is short-circuited.

## See also

- [Extensions](extensions.md)
- [API Reference](../api/reference.md)
