# Configs

GeneLab's config system is a small dataclass hierarchy under `genelab.configs`:

```
TaskCfg
└── env: object   # downstream extensions plug in their own env dataclass
    ManagerBasedEnvCfg
    ├── scene
    ├── actions
    ├── observations
    ├── rewards
    ├── terminations
    └── events
```

`TaskCfg.env` is typed `object` on purpose — downstream extensions plug their own env dataclass
into it without touching core. Most users will subclass `ManagerBasedEnvCfg` for their `env` field.

## apply_overrides

The headline function is `apply_overrides(cfg, dict)`, which parses dotted paths and applies them
to the config tree:

```python
from genelab.configs import apply_overrides

apply_overrides(cfg, {
    "env.scene.dt": "0.005",
    "env.scene.steps": "500",
    "env.actions.scale": "0.3",
    "env.observations.include_velocity": "true",
})
```

### Type coercion

Strings are coerced using the field's type hint on the target dataclass. Supported targets:

| Hint | Accepted strings |
|------|------------------|
| `bool` | `true`/`false`/`1`/`0`/`yes`/`no` (case-insensitive) |
| `int` | any base-10 integer literal |
| `float` | any Python float literal |
| `Path` | passed through `pathlib.Path(...)` |
| `list[T]` | comma-separated, each element coerced as `T` |
| `tuple[T, ...]` | comma-separated, each element coerced as `T` |
| `str` | identity |

If a path doesn't resolve to a known field or the value can't be coerced, `apply_overrides` raises
a descriptive error early — failures surface at config build time, not at simulation runtime.

### CLI integration

The `play` / `train` subcommands forward every `--<a.b.c> VALUE` flag into `apply_overrides`. The
three short flags (`--vis`, `--gpu`, `--steps`) rewrite to `env.scene.{vis,gpu,steps}` overrides
before forwarding. See [Play and Train](../cli/play-train.md).

## API

The full auto-generated reference for `genelab.configs` is on the
[API Reference](../api/reference.md) page.
