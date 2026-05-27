# Configuration Reference

GeneLab configuration is plain Python dataclasses. The CLI mutates these dataclasses before runtime
through dotted overrides; task code can construct or clone them directly.

## Top-level task shape

| Field | Type | Meaning |
|---|---|---|
| `TaskCfg.name` | `str` | Stable task id registered in `TASKS`. |
| `TaskCfg.env_name` | `str` | Name of the registered environment. |
| `TaskCfg.robot_name` | `str` | Name of the registered robot. |
| `TaskCfg.env` | `object` | Training/default environment config. Usually a `ManagerBasedRlEnvCfg` subclass. |
| `TaskCfg.play_env` | `object | None` | Optional play-mode config. `genelab play` prefers it when present. |
| `TaskCfg.agent` | `object | None` | Runner config. RSL-RL tasks use `RslRlOnPolicyRunnerCfg`. |
| `TaskCfg.trainable` | `bool` | Informational flag shown by registry views. |

## Runtime configuration

| Dataclass | Important fields |
|---|---|
| `SimulationCfg` | `vis`, `gpu`, `steps`, `dt`, `substeps`, `num_envs` |
| `InteractiveSceneCfg` | `env_spacing`, `sensors`, `mouse_interaction`, `entities`, `terrain`, `batch_render`, `recordings` |
| `ManagerBasedEnvCfg` | `device`, `simulation`, `scene`, and lightweight enabled flags for core manager groups |
| `ManagerBasedRlEnvCfg` | `decimation`, `episode_length_s`, `seed`, `robot`, `actions_cfg`, `observations_cfg`, `rewards_cfg`, `terminations_cfg`, `commands_cfg`, `events_cfg`, `curriculum_cfg`, `metrics_cfg`, `bridges_cfg` |

## Override grammar

Any `--a.b.c VALUE` after a `play` or `train` task id becomes an override. Hyphens in option names
are converted to underscores.

```bash
genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

The path is resolved against `TaskCfg`, so `env.*` points to `task.cfg.env` and `play_env.*` points
to `task.cfg.play_env`.

## Shortcut aliases

| CLI flag | Play target | Train target |
|---|---|---|
| `--vis` / `-v` | `play_env.simulation.vis=true` when `play_env` exists, else `env.simulation.vis=true` | `env.simulation.vis=true` |
| `--gpu` | `play_env.simulation.gpu=true` when `play_env` exists, else `env.simulation.gpu=true` | `env.simulation.gpu=true` |
| `--steps N` | `play_env.simulation.steps=N` when `play_env` exists, else `env.simulation.steps=N` | `--max_iterations N` |
| `--dt X` | `play_env.simulation.dt=X` when `play_env` exists, else `env.simulation.dt=X` | `env.simulation.dt=X` |

## Type coercion

`apply_overrides` reads dataclass annotations when available and converts strings to the target
type.

| Target type | Accepted value |
|---|---|
| `bool` | `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off` |
| `int` | Decimal integer |
| `float` | Python float literal |
| `Path` | `pathlib.Path(value)` |
| `list[T]` | Comma-separated values coerced as `T` |
| `tuple[T, ...]` | Comma-separated values coerced as `T` |
| `None` | `none` or `null` |
| Other | String fallback |

Unknown paths and invalid conversions fail before the simulator starts.

## Discovering paths

```bash
genelab info GeneLab-Inverted-Pendulum-v0
```

Copy paths from `Overridable cfg paths`. This is safer than guessing nested term names by hand.
