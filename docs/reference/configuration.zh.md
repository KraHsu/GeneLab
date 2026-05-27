# 配置参考

GeneLab 配置是普通 Python dataclass。CLI 在运行前通过 dotted override 修改这些 dataclass；
任务代码也可以直接构造或复制它们。

## 顶层任务结构

| 字段 | 类型 | 含义 |
|---|---|---|
| `TaskCfg.name` | `str` | 注册到 `TASKS` 的稳定 task id。 |
| `TaskCfg.env_name` | `str` | 已注册环境名。 |
| `TaskCfg.robot_name` | `str` | 已注册机器人名。 |
| `TaskCfg.env` | `object` | 默认/训练环境配置，通常是 `ManagerBasedRlEnvCfg` 子类。 |
| `TaskCfg.play_env` | `object | None` | 可选的 play 模式配置。`genelab play` 优先使用它。 |
| `TaskCfg.agent` | `object | None` | runner 配置。RSL-RL 任务使用 `RslRlOnPolicyRunnerCfg`。 |
| `TaskCfg.trainable` | `bool` | 注册表视图展示用标志。 |

## 运行时配置

| Dataclass | 重要字段 |
|---|---|
| `SimulationCfg` | `vis`、`gpu`、`steps`、`dt`、`substeps`、`num_envs` |
| `InteractiveSceneCfg` | `env_spacing`、`sensors`、`mouse_interaction`、`entities`、`terrain`、`batch_render`、`recordings` |
| `ManagerBasedEnvCfg` | `device`、`simulation`、`scene` 和轻量 manager enabled flag |
| `ManagerBasedRlEnvCfg` | `decimation`、`episode_length_s`、`seed`、`robot`、`actions_cfg`、`observations_cfg`、`rewards_cfg`、`terminations_cfg`、`commands_cfg`、`events_cfg`、`curriculum_cfg`、`metrics_cfg`、`bridges_cfg` |

## Override 语法

`play` 或 `train` 的 task id 后面，任何 `--a.b.c VALUE` 都会变成 override。选项名里的
连字符会转换成下划线。

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

路径从 `TaskCfg` 开始解析，因此 `env.*` 指向 `task.cfg.env`，`play_env.*` 指向
`task.cfg.play_env`。

## 短标志别名

| CLI 标志 | Play 目标 | Train 目标 |
|---|---|---|
| `--vis` / `-v` | 有 `play_env` 时为 `play_env.simulation.vis=true`，否则为 `env.simulation.vis=true` | `env.simulation.vis=true` |
| `--gpu` | 有 `play_env` 时为 `play_env.simulation.gpu=true`，否则为 `env.simulation.gpu=true` | `env.simulation.gpu=true` |
| `--steps N` | 有 `play_env` 时为 `play_env.simulation.steps=N`，否则为 `env.simulation.steps=N` | `--max_iterations N` |
| `--dt X` | 有 `play_env` 时为 `play_env.simulation.dt=X`，否则为 `env.simulation.dt=X` | `env.simulation.dt=X` |

## 类型转换

`apply_overrides` 会读取 dataclass 注解，并把字符串转换到目标类型。

| 目标类型 | 接受值 |
|---|---|
| `bool` | `true`、`false`、`1`、`0`、`yes`、`no`、`on`、`off` |
| `int` | 十进制整数 |
| `float` | Python float 字面量 |
| `Path` | `pathlib.Path(value)` |
| `list[T]` | 逗号分隔，并按 `T` 转换 |
| `tuple[T, ...]` | 逗号分隔，并按 `T` 转换 |
| `None` | `none` 或 `null` |
| 其他 | 字符串兜底 |

未知路径和无法转换的值会在仿真启动前报错。

## 发现可覆盖路径

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

从 `Overridable cfg paths` 里复制路径，比手写嵌套 term 名更可靠。
