# Configs

The config system is a small dataclass hierarchy under `genelab.configs`:

```
TaskCfg
└── env: object   # downstream extensions plug in their own env dataclass
    ManagerBasedEnvCfg
    ├── simulation
    ├── scene
    ├── actions
    ├── observations
    ├── rewards
    ├── terminations
    └── events
```

`TaskCfg.env` is typed `object` on purpose — downstream extensions plug their own env dataclass
into it without touching core. Most users subclass `ManagerBasedEnvCfg` for their `env` field.

## apply_overrides

The headline function is `apply_overrides(cfg, dict)`, which parses dotted paths and applies
them to the config tree:

```python
from genelab.configs import apply_overrides

apply_overrides(cfg, {
    "env.simulation.dt": "0.005",
    "env.simulation.steps": "500",
    "env.robot.actuators.cart.stiffness": "100.0",
    "env.observations.include_velocity": "true",
})
```

### Type coercion

String values are coerced using the field's type hint on the target dataclass. Supported
targets:

| Hint | Accepted strings |
|------|------------------|
| `bool` | `true`/`false`/`1`/`0`/`yes`/`no` (case-insensitive) |
| `int` | any base-10 integer literal |
| `float` | any Python float literal |
| `Path` | passed through `pathlib.Path(...)` |
| `list[T]` | comma-separated, each element coerced as `T` |
| `tuple[T, ...]` | comma-separated, each element coerced as `T` |
| `str` | identity |

When a path does not resolve to a known field or a value cannot be coerced, `apply_overrides`
raises a descriptive error at config build time, not at simulation runtime.

### Forwarding from the CLI

The `play` / `train` subcommands forward every `--<a.b.c> VALUE` flag into `apply_overrides`.
The three simulation shortcuts (`--vis`, `--gpu`, `--steps`) rewrite to
`env.simulation.{vis,gpu,steps}` overrides before forwarding.

## See also

- [Actuators](actuators.md)
- [Play and Train](../cli/play-train.md)
- [API Reference](../api/reference.md)
