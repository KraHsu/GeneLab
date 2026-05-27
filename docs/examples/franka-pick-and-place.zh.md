# Franka 抓取放置

针对 asset-zoo Franka Panda 的 goal-conditioned manipulation 任务，使用
Stable-Baselines3 SAC + Hindsight Experience Replay 训练。一个 4 cm 立方体在
机械臂前方生成，每个环境采样的目标位置可能在地面，也可能在空中（与
panda-gym `PandaPickAndPlace` 分布一致）。

## 任务

| Task id | Action dim | 算法 |
|---|---:|---|
| `GeneLab-Franka-Pick-And-Place-v0` | 4 | SB3 SAC + HER + lift bonus + FSM demo prefill |

Action 向量为 `(dx, dy, dz, gripper)`——七个手臂关节上的
`DifferentialIKAction`（朝向锁定为 panda-gym 朝下姿态），加上手指关节上的
`ContinuousGripperAction`。

## 安装扩展

```bash
uv pip install -e examples/franka_pick_and_place
genelab list tasks | grep Franka
```

不安装时：

```bash
PYTHONPATH=examples/franka_pick_and_place/src \
  genelab --import genelab_franka_pick_and_place.tasks list tasks
```

第一次运行会下载 Franka MJCF 资产并构建 Genesis kernel cache。

## Smoke 训练

```bash
genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 16 --max-iterations 2000
```

## 完整训练

HER 单独无法学会抬起方块——sparse goal reward 在空中目标区域没有任何学习信号。
两个组件共同打破这个平台，缺一不可：

1. **Lift bonus** 进入 env reward（`mdp.lift_bonus`，权重 `0.2`）：每步从 `0`
   （方块在桌上）线性升到 `+0.2`（方块距桌面 10 cm 以上），并在 HER 的
   compute-reward 回调中完全镜像，让 in-buffer 与 relabelled reward 形状一致。
2. **FSM demo prefill**：`demo_fsm.py` 中的手写控制器产生完整的
   `reach → grasp → lift → place` 轨迹，`collect_demos.py` 把这些 transition
   存成 `.npz`；SB3 backend 在 `model.learn` 之前根据 `GENELAB_SB3_DEMO_PATH`
   （或 `Sb3AgentCfg.demo_path`）把这些 transition 重放进 replay buffer。

```bash
# 1. 收集 demo（num-envs 32 大约一分钟）。
python -m genelab_franka_pick_and_place.collect_demos \
  --num-envs 32 --steps 6400 --out /tmp/franka_pp_demos.npz

# 2. 带 demo prefill 训练。
GENELAB_SB3_DEMO_PATH=/tmp/franka_pp_demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
```

预期 success-rate 曲线节奏：

| step bucket | mean success | 发生了什么 |
|---|---|---|
| 0–200 K     | 5–7 %   | demo 在 buffer 里，SAC 还没把它消化进策略 |
| 200 K–400 K | 10–20 % | 策略开始模仿抬起动作 |
| 400 K–600 K | 30–40 % | 突破桌面 scoot 的天花板 |
| 600 K–1 M   | 60–85 % | 任务基本攻克 |
| 1.4 M+      | 95 %+   | 收敛平台，单点 peak 约 99 % |

## 回放 checkpoint

```bash
genelab play GeneLab-Franka-Pick-And-Place-v0 \
  --checkpoint logs/sb3/franka_pick_and_place/<run>/model.zip \
  --steps 200
```

## 物理参数覆盖

env config 在 asset-zoo Franka 默认值上做了三处调整，作用域限定本任务：

| 组件 | 默认值 | 覆盖值 | 原因 |
|---|---|---|---|
| `panda_arm.stiffness` / `damping` | 400 / 80 | **2000 / 200** | 默认 PD 在抓住方块后会下垂到无法恢复的高度；更硬的 PD 才能抵抗重力与方块负载。 |
| `panda_hand.effort_limit` / `velocity_limit` | 20 / 0.2 | **100 / 1.0** | 默认值闭合速度约 0.1 rad/s——比方块从张开爪子掉下来还慢。新限值让一个 env step 内就能合上爪子。 |
| 方块 `friction` | Genesis 默认 | **1.0** | Genesis 内置 rigid friction 太低，无法把方块卡在两指之间。与 panda-gym MuJoCo 方块一致。 |

## 代码入口

| 文件 | 作用 |
|---|---|
| `tasks.py` | 注册 robot、env 和 task id。 |
| `env_cfg.py` | 构建场景、observation group、reward term、物理覆盖。 |
| `mdp.py` | 任务自有的 observation、reward、termination、reset 事件。 |
| `sb3_cfg.py` | SAC + HER agent config 与 HER `compute_reward` 回调。 |
| `demo_fsm.py` | 手写 FSM，产生 `reach → grasp → lift → place` 演示。 |
| `collect_demos.py` | CLI 脚本，让 FSM 在 env 中跑，把 transition 存为 `.npz`。 |
| `robot.py` | 包装 asset-zoo Franka，提供 panda-gym 中位姿和 Jacobian/IK 开关。 |

## 说明

- 方块直接放在 ground plane 上；场景里没有单独的桌子 mesh。
- Goal `z` 有 `0.7` 的概率在 `[0, 0.2]` 中均匀采样；否则目标位于地面上的方块高度。
- 成功距离阈值是 `0.05 m`。

## 另见

- [MDP term 参考](../concepts/mdp.md)
- [设计任务](../best-practices/task-design.md)
- [资产库](../concepts/asset_zoo.md)
