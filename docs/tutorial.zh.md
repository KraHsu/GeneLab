---
hide:
  - toc
---

# 用 GeneLab 构建第一个机器人实验

本教程从一个干净 checkout 开始，带你完成 GeneLab 的最小闭环：安装环境、查看注册表、运行
倒立摆、理解任务如何接入 Genesis 后端、修改配置、启动一次短训练，最后创建一个自己的下游
扩展项目。教程末尾会把同一套结构延伸到 Unitree G1，作为完整机器人工作流的进阶入口。完成后，
你应该能说清 GeneLab 的整体架构，并知道下一步去哪里查模块、API 和最佳实践。

目标读者是熟悉 Python 与命令行、但第一次接触 GeneLab 的机器人或强化学习开发者。教程不要求
你先理解所有内部模块；每一步都会给出可验证结果。

## 你会构建什么

你会运行仓库自带的 `GeneLab-Inverted-Pendulum-v0` 任务。这个任务很小，但覆盖了 GeneLab
的核心路径：

```text
CLI
└── TASKS registry
    └── TaskCfg
        ├── env / play_env: ManagerBasedRlEnvCfg
        ├── robot: ArticulationCfg
        ├── manager terms: actions, observations, rewards, terminations, events
        └── agent: RslRlOnPolicyRunnerCfg
            └── Genesis-backed ManagerBasedRlEnv
```

后续复杂任务，例如 Unitree G1、地形课程、传感器和数据录制，本质上是在这条路径上增加更大的
机器人、更丰富的 manager term，以及更完整的 runner 配置。

## 1. 准备环境

在仓库根目录同步依赖：

```bash
uv sync --extra torch-cpu
uv run genelab --version
```

如果你有 NVIDIA GPU，把 `torch-cpu` 换成与你驱动匹配的 `torch-cu126`、`torch-cu128` 或
`torch-cu130`。每次只能选一个 `torch-*` extra。

你应该看到类似输出：

```text
genelab 0.1.0
```

初始化 Genesis 和绘图库需要的项目本地缓存：

```bash
uv run genelab cache
```

如果安装失败或 torch 版本不对，先看 [安装](getting-started/installation.md)。

## 2. 查看 GeneLab 当前知道什么

GeneLab 的核心包只提供框架。机器人、环境和任务由内置资产库或扩展包注册进来。先列出三张
注册表：

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

你应该能看到内置资产库的机器人，以及已通过 entry point 发现的示例任务。若任务列表为空，
说明示例扩展没有被安装或发现；可以显式导入倒立摆扩展：

```bash
uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

注册表是 GeneLab 的第一层架构边界：

| 注册表 | 放什么 | 典型消费者 |
|---|---|---|
| `ROBOTS` | 机器人或资产配置 factory | 环境配置、CLI `list robots` |
| `ENVS` | 可构造环境的 factory | CLI `list envs`、下游项目 |
| `TASKS` | 包含 env、play env、agent 的任务 factory | `genelab play`、`genelab train` |

## 3. 运行第一个任务

先用短 rollout 验证任务可以构造并走完几步：

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --steps 32
```

如果你在本地桌面环境中运行，可以打开 Genesis viewer：

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
```

这一步经过的路径是：

1. CLI 加载 entry-point 扩展和显式 `--import` 扩展。
2. CLI 在 `TASKS` 中查找 `GeneLab-Inverted-Pendulum-v0`。
3. task 返回 `TaskCfg`，其中 `play_env` 是可视化友好的环境配置。
4. `ManagerBasedRlEnv` 构造 Genesis scene、机器人 articulation、传感器和各类 manager。
5. play loop 产生动作并调用 `env.step(action)`。

## 4. 看清一个任务的可改配置

用 `info` 查看任务元信息和可覆盖路径：

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

重点看输出里的 `Overridable cfg paths`。CLI 的每个 `--a.b.c VALUE` 都会落到
`TaskCfg` 的某个 dataclass 字段上，并由 `apply_overrides` 自动转换类型。

试着改仿真步长和 episode 步数：

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 \
  --steps 128 \
  --env.simulation.dt 0.005
```

常用短标志会被改写为配置路径。`play` 中，如果 task 定义了 `play_env`，短标志会优先作用到
`play_env`；否则才作用到 `env`。

| 短标志 | 有 `play_env` 时的 play 目标 | train 目标 |
|---|---|---|
| `--vis` | `play_env.simulation.vis=true` | `env.simulation.vis=true` |
| `--gpu` | `play_env.simulation.gpu=true` | `env.simulation.gpu=true` |
| `--steps N` | `play_env.simulation.steps=N` | `--max_iterations N` |
| `--dt X` | `play_env.simulation.dt=X` | `env.simulation.dt=X` |

## 5. 打开任务源码并对照架构

倒立摆的注册入口在 `examples/inverted_pendulum/src/genelab_inverted_pendulum/tasks.py`。
它做三件事：

```python
register_robot("inverted-pendulum", get_inverted_pendulum_robot_cfg, ...)
register_env("inverted-pendulum-env", lambda: ManagerBasedRlEnv(...), ...)
register_task("GeneLab-Inverted-Pendulum-v0", InvertedPendulumTask, ...)
```

环境配置在
`examples/inverted_pendulum/src/genelab_inverted_pendulum/single/env_cfg.py`。这里能看到
GeneLab 的 manager-based MDP 形状：

```python
actions_cfg={
    "cart": JointPositionActionCfg(...)
}
observations_cfg={
    "policy": ObservationGroupCfg(terms={...}),
    "critic": ObservationGroupCfg(terms={...}),
}
rewards_cfg={
    "alive": RewardTermCfg(...),
    "pole_upright": RewardTermCfg(...),
}
terminations_cfg={
    "time_out": TerminationTermCfg(...),
    "pole_fell": TerminationTermCfg(...),
}
events_cfg={
    "reset_joints": EventTermCfg(mode="reset", ...)
}
```

这一层是 GeneLab 最重要的能力：把强化学习环境拆成可替换的 term，而不是把所有逻辑塞进一个
巨大的 `step()` 函数。

## 6. 启动一次短训练

倒立摆任务自带 RSL-RL runner 配置。先跑一个很短的训练，验证 RL 管线可以构造 env、包装
VecEnv、写日志和 checkpoint：

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

训练结束后，查看日志目录：

```bash
ls logs/rsl_rl
```

你会看到按实验名和时间戳组织的目录。主进程会写入：

```text
params/env.json
params/agent.json
model_*.pt
```

回放训练好的策略时传入 checkpoint：

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 \
  --agent trained \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_*.pt
```

如果只是检查动作边界或环境稳定性，也可以用 `--agent zero` 或 `--agent random`。

## 7. 创建自己的扩展项目

GeneLab 推荐把真实项目放在独立 Python 包里，而不是直接改 `src/genelab/`。生成一个项目骨架：

```bash
uv run genelab project new my_robot_project
```

骨架会包含：

```text
my_robot_project/
├── pyproject.toml
└── src/my_robot_project/
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

关键是 `pyproject.toml` 里的 entry point：

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project.tasks:register"
```

安装为 editable 包之后，CLI 会自动发现它：

```bash
uv pip install -e my_robot_project
uv run genelab list tasks
```

如果暂时不安装，也可以显式导入：

```bash
PYTHONPATH=my_robot_project/src \
  uv run genelab --import my_robot_project.tasks list tasks
```

## 8. 继续到 Unitree G1

Unitree G1 示例使用同一套架构，但换成了人形机器人、更大的 action/observation 空间、
命令采样、RSL-RL 训练和动作模仿。

安装扩展：

```bash
uv pip install -e examples/unitree
uv run genelab list tasks
```

你应该能看到：

```text
Genelab-Velocity-Flat-Unitree-G1-v0
Genelab-Tracking-Flat-Unitree-G1-v0
```

训练前先做可视化 smoke test：

```bash
uv run genelab play Genelab-Velocity-Flat-Unitree-G1-v0 --vis --steps 500
```

两个 Unitree 任务展示 GeneLab 的进阶能力：

| Task | 展示能力 |
|---|---|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | 速度命令、人形机器人 action scale、PPO 训练、checkpoint 回放。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | motion command 加载、参考动作回放、body pose tracking reward。 |

完整训练与回放命令见 [Unitree G1](examples/unitree-g1.md)。

## 9. GeneLab 能力地图

到这里，你已经走完最小闭环，也看到了更大机器人示例的位置。GeneLab 的能力可以按四层理解：

| 层级 | 模块 | 作用 |
|---|---|---|
| 发现与调度 | `genelab.registry`、`genelab.cli` | 注册机器人、环境、任务；用 CLI 发现、运行、训练 |
| 配置与 MDP | `genelab.configs`、`genelab.managers`、`genelab.mdp` | dataclass 配置、override、action/obs/reward/termination/event/curriculum term |
| 仿真对象 | `genelab.scene`、`genelab.entity`、`genelab.actuator`、`genelab.sensor`、`genelab.terrains` | Genesis scene、articulation、刚体、执行器、传感器、地形 |
| 训练与实验 | `genelab.rl`、`genelab.recording`、`genelab.bridges` | RSL-RL 训练/回放、数据录制、实时绘图、键盘或 GUI 控制 |

## 下一步

按目标继续阅读：

- 想跑更多命令：看 [CLI 总览](cli/overview.md) 和 [play 与 train](cli/play-train.md)。
- 想理解架构：看 [模块地图](reference/module-map.md)、[Manager 与 MDP term](concepts/managers.md)。
- 想写自己的任务：看 [设计任务配置](best-practices/task-design.md) 和
  [扩展加载](concepts/extensions.md)。
- 想查 API：看 [API 参考](api/reference.md)。
- 想看完整示例：看 [示例](examples/overview.md)。
