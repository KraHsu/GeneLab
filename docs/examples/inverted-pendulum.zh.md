# 倒立摆

`examples/inverted_pendulum/` 提供两个在平面上的 cart-pole 经典控制 PPO 任务。整条训练栈与
`examples/unitree/` 对齐：基于 Genesis 的 `ManagerBasedRlEnv`、rsl_rl PPO、挂在杆上的
`BodyVelocitySensor`，以及统一的 `genelab train` / `genelab play` CLI。

## 任务列表

| Task id | 问题 |
|---------|------|
| `GeneLab-Inverted-Pendulum-v0` | 小车 + 单杆的倒立摆。 |
| `GeneLab-Double-Inverted-Pendulum-v0` | 小车 + 串联双杆的倒立摆。 |

## 安装

扩展依赖 `rl` extra（rsl_rl）。`torch-*` extra 按硬件挑选。

```bash
uv sync --extra rl --extra torch-cu128
uv pip install -e examples/inverted_pendulum

uv run genelab list tasks
# -> GeneLab-Inverted-Pendulum-v0
# -> GeneLab-Double-Inverted-Pendulum-v0
```

## 单倒立摆

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
    --num-envs 4096 --max-iterations 150

uv run genelab play  GeneLab-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/inverted_pendulum_flat/<run>/model_150.pt --vis
```

传入 `--checkpoint` 会让 `play` 自动经过 RL runner，并默认使用 `--agent trained`。

## 双倒立摆

```bash
uv run genelab train GeneLab-Double-Inverted-Pendulum-v0 \
    --num-envs 4096 --max-iterations 300

uv run genelab play  GeneLab-Double-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/double_inverted_pendulum_flat/<run>/model_300.pt --vis
```

## 传感器与欠驱动

只有小车的 slide 关节通过 PD 控制。两个 pole hinge 默认 `kp=0, kv=0`，保证整体处于欠驱动状态，
策略必须通过小车水平运动间接稳定杆。顶端 pole 上挂载的 `BodyVelocitySensor` 给出一路带噪声的
角速度观测：policy 观测组用 `Unoise` 做 corruption，critic 观测组直接读取干净值。

## 交互式扰动

`play` 默认只开 1 个环境（`num_envs=1`），并启用 Genesis 的 `MouseInteractionPlugin`。
左键点击 cart 或 pole 并拖动，会有一根弹簧把所点击的 link 拉向光标位置；策略仍然在背后试图
保持平衡。滚轮可绕表面法线旋转拖拽平面，松开左键即移除外力。

!!! tip "Smoke-test 预算"
    使用 `--num-envs 64 --max-iterations 5` 跑 5–10 次迭代足以验证整条链路。此时 reward 信号
    仍非常嘈杂，真正收敛需要上面给出的 150 / 300 次迭代预算。

## 日志

两个任务都把日志写到 `logs/rsl_rl/<experiment>/<timestamp>_/`，结构与 Unitree 示例一致：

- `params/env.json` 与 `params/agent.json` —— 运行时冻结的配置快照。
- `model_<iter>.pt` —— 按 `save_interval` 保存的 checkpoint。
- 同目录下的 TensorBoard 事件文件。

## See also

- [Unitree G1 快速开始](../getting-started/quickstart.md#unitree-g1)
- [传感器](../concepts/sensors.md)
- [play 与 train CLI](../cli/play-train.md)
