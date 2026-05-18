# 魔方

`examples/genelab_examples/src/genelab_examples/rubiks/` 是一个 play-only 的 Genesis 演示，
模拟一个力驱动的 3×3 魔方。这个任务**不是** RL 任务——它绕开 `ManagerBasedRlEnv`，
通过一个自定义控制器直接驱动 Genesis `Scene`。它存在的意义在于演示：(1) 如何在 GeneLab 注册表
中注册一个非 RL 任务；(2) 如何在运行时即时生成 MJCF 资产；(3) 如何用 Genesis 的动态约束 API
让一个刚体在不同关节形态之间切换。

## 任务列表

| Task id | 问题 |
|---------|------|
| `GeneLab-Rubiks-Play-v0` | 27 个 cubie 起初被动态焊成一整块刚体；一旦绕某根世界轴的角速度足够大，便沿该轴拆成三层平板，每层之间用 hinge 约束相连。 |

## 安装

```bash
uv sync --extra torch-cu128
uv pip install -e examples/genelab_examples

uv run genelab list tasks | grep Rubiks
# -> GeneLab-Rubiks-Play-v0
```

或者把 `src` 加到 `PYTHONPATH` 上免安装跑：

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks list tasks
```

## 运行

```bash
uv run genelab play GeneLab-Rubiks-Play-v0 --vis --steps 600

# 更大的 cubie、更小的间隙。
uv run genelab play GeneLab-Rubiks-Play-v0 --steps 5 \
    --env.robot.cubie_size 0.04 --env.robot.gap 0.002

# 把每个 cubie 静态焊死（用于和默认模式对比）。
uv run genelab play GeneLab-Rubiks-Play-v0 --env.robot.welded true --steps 5

# 鼠标拖拽交互（需要 --vis）。
uv run genelab play GeneLab-Rubiks-Play-v0 --vis --steps 0 \
    --env.interaction.interactive_force true
```

`--steps 0` 让 viewer 循环到窗口被关闭为止。不传 `--steps` 时使用
`SimulationCfg.steps` 的默认值。

## 这个示例演示了什么

| GeneLab 能力 | 出现位置 | 概念文档 |
|---|---|---|
| 通过 `register_task` + `TaskCfg` 注册任务 | `tasks.py` | [Registry](../concepts/registry.md) |
| 通过 `register_robot` 注册机器人（MJCF 由 `RubiksCubeSpec` 即时写出） | `robots.py`、`rubiks/assets.py:write_mjcf` | [Registry](../concepts/registry.md)、[Asset zoo](../concepts/asset_zoo.md) |
| 配置 dataclass 模式（`RubiksEnvCfg` 继承 `ManagerBasedEnvCfg`） | `rubiks/config.py` | [Configs](../concepts/configs.md) |
| 自定义 `play()` runner，完全绕过 manager pipeline | `envs.py:RubiksPlayEnv.play` | 对比 [Manager 与 MDP term](../concepts/managers.md) |
| Genesis 动态约束 API（weld / hinge / 外加力矩） | `rubiks/sim.py:ForceDrivenCubeController` | [场景与实体](../concepts/scene.md) |
| Genesis `MouseInteractionPlugin` 实时拖拽 | `envs.py:RubiksPlayEnv.play` | 与[倒立摆](inverted-pendulum.md)使用同一插件 |

## 代码走读

每个文件只负责一件事：

- **`rubiks/config.py`（49 行）** —— `RubiksEnvCfg`、`RubiksRobotCfg`、
  `RubiksInteractionCfg`、`ForceDrivenCubeConfig`，全部可以通过点号 CLI flag 覆写
  （`--env.robot.cubie_size 0.04`、`--env.interaction.mouse_spring 25` 等）。
- **`rubiks/assets.py`（249 行）** —— 纯几何。`RubiksCubeSpec` 参数化 cubie 尺寸、贴纸内缩、
  摩擦、质量等。`iter_cubie_coords` 与 `exposed_faces` 枚举 27 个 cubie。`write_mjcf`
  生成 Genesis 加载的 MJCF。
- **`rubiks/sim.py`（1061 行）** —— 示例的主要逻辑所在。这里有两个控制器：
    - `RubiksCubeController` —— 历史遗留的运动学转动器，靠 `set_qpos` 直接改层位姿，
      在 `legacy_turn.enabled=true` 时作为视觉参照保留。
    - `ForceDrivenCubeController` —— 默认控制器，一个状态机：
        1. 用 `solver.add_weld_constraint` 把 27 个 cubie 焊成一整块。
        2. 监测整块的角速度，发现 `|ω| > enter_ang_vel` 沿某条轴超过阈值时，
           **只**沿该轴拆掉对应焊点，并加上 hinge-constraint 对，使 cube 拆成三层平板。
        3. 通过 `solver.apply_links_external_torque` 施加一个
           `joint_spring + joint_damping` 弹簧+阻尼力矩，把边层拉到最近的四分之一圈位置。
        4. 当边层角度小于 `exit_angle` **且**角速度小于 `exit_ang_vel`，就拆掉 hinge、
           重新焊回。
    - 这部分代码长是因为本质复杂：27 个 cubie × 3 根轴 × 2 个方向 × 多步稳定逻辑，
      每次状态切换都要做 Genesis 约束簿记。删掉任何一段都会改变行为，并非"水分"。
- **`envs.py:RubiksPlayEnv`** —— 构造 Genesis `Scene`（针对大量约束调优 rigid options：
  `max_dynamic_constraints=128`、`noslip_iterations=5`），加载生成的 MJCF，按需挂上
  `MouseInteractionPlugin`，然后跑 `cfg.simulation.steps` 步控制器循环。**没有
  `ManagerBasedRlEnv` 参与**——这个任务没有 obs、action、reward。
- **`tasks.py`** —— 把 `GeneLab-Rubiks-Play-v0` 注册为 play-only 任务
  （`trainable=False`），对它调用 `genelab train` 会直接抛 `NotImplementedError`。

## Smoke test

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks play GeneLab-Rubiks-Play-v0 --steps 5
```

5 步就足够验证资产生成、场景构造和一步控制器逻辑。首次运行会编译 Genesis kernel，
比较慢，属于正常现象。

## See also

- [五指手](wuji-hand.md) —— 同一个 extension 里另一个 play-only 示例。
- [Registry](../concepts/registry.md) —— `register_task` / `register_robot` 的工作机制。
- [Configs](../concepts/configs.md) —— `--env.robot.welded true` 走的 dataclass 覆写机制。
- [场景与实体](../concepts/scene.md) —— 控制器用到的 Genesis 原语。
- [倒立摆](inverted-pendulum.md) —— 另一个用 `MouseInteractionPlugin` 的示例。
