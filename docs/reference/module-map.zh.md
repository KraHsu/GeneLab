# 模块地图

这是一页查阅型参考，用来把 GeneLab 的公开模块映射到它们负责的系统边界。你知道要改什么、但不确定从哪里导入时，先看这里。

## 公共 facade

| 模块 | 用途 | 常见导入 |
|---|---|---|
| `genelab.lab` | 稳定的用户入口，汇总配置、注册表、scene 对象、传感器、执行器、地形和扩展加载。 | `TaskCfg`、`ManagerBasedEnvCfg`、`ArticulationCfg`、`SensorCfg` |
| `genelab` | 轻量包根。懒加载少量顶层类型，避免导入包根时立刻拉起 torch。 | `__version__`、`TaskCfg`、`ManagerBasedEnvCfg` |

下游任务、notebook 和示例优先从 `genelab.lab` 导入；只有需要明确的低层能力时再进入对应模块。

## 发现与调度

| 模块 | 职责 |
|---|---|
| `genelab.registry` | `ROBOTS`、`ENVS`、`TASKS`、`RegistryEntry`、注册 helper、entry point 加载、显式扩展导入。 |
| `genelab.cli` | Typer/Rich 命令行：`cache`、`list`、`info`、`play`、`train`、`prof`、`project new`。 |
| `genelab.cache` | 创建 Genesis、Quadrants、Matplotlib 使用的项目本地缓存目录。 |

## 配置与 MDP

| 模块 | 职责 |
|---|---|
| `genelab.configs` | 核心 dataclass：`SimulationCfg`、`InteractiveSceneCfg`、`ManagerBasedEnvCfg`、`TaskCfg` 和 `apply_overrides`。 |
| `genelab.envs.manager_based_rl_env` | Genesis 后端的 `ManagerBasedRlEnv` 与 `ManagerBasedRlEnvCfg`。 |
| `genelab.managers` | action、command、observation、reward、termination、event、curriculum、metric 的 manager 与 term cfg。 |
| `genelab.mdp` | 可复用的 observation、reward、termination、event、curriculum、metric、noise、action、command、domain randomization 函数。 |

## 仿真对象

| 模块 | 职责 |
|---|---|
| `genelab.scene` | `InteractiveScene`，持有 live Genesis scene。 |
| `genelab.entity` | `Articulation`、`RigidObject` 及其配置。 |
| `genelab.actuator` | 隐式 PD、理想 PD、DC motor 执行器模型。 |
| `genelab.sensor` | 传感器基类与 body velocity、IMU、camera、contact、ray-cast、frame transformer、terrain height、self-contact、angular momentum 传感器。 |
| `genelab.terrains` | 子地形配置、地形网格生成、导入 Genesis。 |
| `genelab.asset_zoo` | 预置机器人资产配置与 motion 资产下载 helper。它是随包内容，不是核心抽象边界。 |

## 训练与实验 I/O

| 模块 | 职责 |
|---|---|
| `genelab.rl` | RSL-RL 配置、VecEnv wrapper、`train_task`、`play_task`、分布式 helper、profiler 集成。 |
| `genelab.recording` | 声明式录制配置：实时曲线、文件、相机视频。 |
| `genelab.bridges` | 运行时输入/控制桥，包括键盘 twist 和 DearPyGui twist 控制。 |
| `genelab.utils` | 数学与资产下载 helper。 |

## 选层规则

| 目标 | 起点 |
|---|---|
| 注册新机器人、环境、任务 | `genelab.registry`，再看 [构建扩展项目](../best-practices/extension-projects.md) |
| 增加 observation、reward、termination | `genelab.managers`、`genelab.mdp`，再看 [Manager 与 MDP term](../concepts/managers.md) |
| 增加传感器 | `genelab.sensor`，再看 [传感器](../concepts/sensors.md) |
| 修改训练或回放 | `genelab.rl`，再看 [运行 RL 实验](../best-practices/rl-experiments.md) |
| 查签名与默认值 | [API 参考](../api/reference.md) |
