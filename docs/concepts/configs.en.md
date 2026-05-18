# Configs

GeneLab configuration is a dataclass tree. The tree is intentionally ordinary Python rather than a
custom schema language: task authors can construct it with normal code, and the CLI can still expose
stable override paths.

## The core idea

`TaskCfg` is the boundary between discovery and runtime. It records the task id, the registered env
and robot names, the default env config, an optional play-mode env config, and an optional agent
config.

```text
TaskCfg
├── env
├── play_env
└── agent
```

`env` is typed as `object` on purpose. Downstream projects can use a subclass such as
`ManagerBasedRlEnvCfg` without changing GeneLab core.

## Why dotted overrides work

The CLI treats `--a.b.c VALUE` as a path through this dataclass tree. `apply_overrides` resolves the
path, reads the current value and annotation, coerces the string value, and mutates the config before
runtime starts.

That keeps experiments reproducible: the final `TaskCfg` can be dumped, and invalid paths fail early
instead of silently being ignored.

## Play configs are first-class

Training and inspection often need different defaults. `play_env` lets a task keep viewer-friendly
settings next to training settings: fewer envs, viewer on, optional mouse interaction, and optional
recording outputs.

## Where to continue

- [Configuration Reference](../reference/configuration.md)
- [Design a Task](../best-practices/task-design.md)
- [CLI Reference](../reference/cli.md)
