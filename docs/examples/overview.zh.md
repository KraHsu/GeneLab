# 示例

仓库在 `examples/` 下提供数个参考扩展，同时也是 CLI 与注册表的集成测试。每个可运行的示例
都有独立的文档页面，本页是入口。

## 能力一览

| 示例 | 演示了什么 | 文档 |
|---|---|---|
| `inverted_pendulum` | `ManagerBasedRlEnv` + PPO、`BodyVelocitySensor`、`RecordingCfg`、`MouseInteractionPlugin` | [倒立摆](inverted-pendulum.md) |
| `unitree` | G1 人形机器人速度跟踪 + 动作模仿、asset zoo 与 motion-clip 管线 | [Unitree G1](unitree-g1.md) |
| `genelab_showcase` | 七个单功能 play 任务（sensor、ray-cast、contact、terrain、curriculum、actuator、recording） | [Showcase](showcase.md) |
| `genelab_examples/rubiks` | 自定义 play runner、运行时生成 MJCF、Genesis 动态约束 API | [魔方](rubiks-cube.md) |
| `genelab_examples/wuji_hand` | MJCF 资产目录、关节名 → DOF 映射、固定轨迹重放 | [五指手](wuji-hand.md) |
| `external_project` | 脚手架模板，与 `genelab project new` 产出形状一致 | 见 [Project New CLI](../cli/project-new.md) |

## inverted_pendulum

两个 PPO cart-pole 任务，训练栈与 Unitree 示例相同（`ManagerBasedRlEnv` + rsl_rl），训练预算
控制在单机能跑完的量级：

- **`GeneLab-Inverted-Pendulum-v0`** —— 小车 + 单杆倒立摆。
- **`GeneLab-Double-Inverted-Pendulum-v0`** —— 小车 + 串联双杆倒立摆。

源码位于 `examples/inverted_pendulum/`。完整走读见 [倒立摆](inverted-pendulum.md)。

## unitree

Unitree G1 人形机器人的两个 PPO 任务 —— 速度跟踪与动作模仿，从 mjlab 移植并适配到 Genesis。
形态与 `genelab_examples` 相同（entry point、`register()`、按模块拆分的注册文件）。源码位于
`examples/unitree/`。完整走读见 [Unitree G1](unitree-g1.md)。

## genelab_showcase

7 个 play-only 任务，每一个对应一个 GeneLab 构建块，把真实 Franka 或 G1 投入一个最小的
`ManagerBasedRlEnv`，并把对应特征的证据落到 `logs/showcase/<slug>/`。面向人眼/数值核对，
不做训练。源码位于 `examples/genelab_showcase/`。完整走读见 [Showcase](showcase.md)。

## genelab_examples

仓库内的标准扩展，接通两个**绕过 `ManagerBasedRlEnv`** 的 play-only Genesis 演示：

- **`GeneLab-Rubiks-Play-v0`** —— 27 cubie 的魔方，通过 Genesis 约束 API 动态焊接 / 重新关节化
  自己。见 [魔方](rubiks-cube.md)。
- **`GeneLab-Wuji-Hand-Playback-v0`** —— Wuji 五指手用 `control_dofs_position` 重放 20 自由度的
  `.npy` 轨迹。见 [五指手](wuji-hand.md)。

`pyproject.toml` 声明了 `genelab.extensions` entry point，因此安装该包后会被自动发现；
项目 `pyproject.toml` 的 `pythonpath` 设置也让测试可以直接 import 而无需安装。源码位于
`examples/genelab_examples/`。

## external_project

下游项目最小模板。`genelab project new` 生成的内容与之结构一致，留在仓库里作为脚手架输出
参考。它**不是**一个可运行的示例任务——使用方式见 [Project New CLI](../cli/project-new.md)
和 [扩展加载](../concepts/extensions.md)。源码位于 `examples/external_project/`。

## See also

- [快速开始](../getting-started/quickstart.md)
- [扩展加载](../concepts/extensions.md)
- [Project New CLI](../cli/project-new.md)
