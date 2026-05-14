# 示例

仓库在 `examples/` 下提供数个参考扩展，同时也是 CLI 与注册表的集成测试。

## inverted_pendulum

两个 PPO cart-pole 任务，训练栈与 Unitree 示例相同（`ManagerBasedRlEnv` + rsl_rl），训练预算
控制在单机能跑完的量级：

- **`GeneLab-Inverted-Pendulum-v0`** —— 小车 + 单杆倒立摆。
- **`GeneLab-Double-Inverted-Pendulum-v0`** —— 小车 + 串联双杆倒立摆。

源码位于 `examples/inverted_pendulum/`；完整流程见 [倒立摆](inverted-pendulum.md)。

## genelab_examples

仓库内的标准扩展，接通两个任务：

- **`wuji_hand`** —— 手部操作任务。
- **`rubiks`** —— 魔方任务。

`pyproject.toml` 声明了 `genelab.extensions` entry point，因此安装该包后会被自动发现；
项目 `pyproject.toml` 的 `pythonpath` 设置也让测试可以直接 import 而无需安装。源码位于
`examples/genelab_examples/`。

## unitree

Unitree G1 人形机器人的两个 PPO 任务 —— 速度跟踪与动作模仿，从 mjlab 移植并适配到 Genesis。
形态与 `genelab_examples` 相同（entry point、`register()`、按模块拆分的注册文件）。源码位于
`examples/unitree/`。

完整动手教程（安装、训练、checkpoint 回放、动作模仿）见
[快速开始 §5](../getting-started/quickstart.md#unitree-g1)。

## external_project

下游项目最小模板。`genelab project new` 生成的内容与之结构一致，留在仓库里作为脚手架输出
参考。源码位于 `examples/external_project/`。

## See also

- [快速开始](../getting-started/quickstart.md)
- [扩展加载](../concepts/extensions.md)
