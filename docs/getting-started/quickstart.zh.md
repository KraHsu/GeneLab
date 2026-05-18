# 快速开始

## 1. 列出已注册内容

核心 `genelab` 包自带空注册表 —— 机器人、环境、任务都由扩展包贡献。仓库内
`examples/genelab_examples/` 通过 `pyproject.toml` 中的 `genelab.extensions` entry point
被自动发现。

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

注册表为空意味着没有扩展被导入。可通过 `uv pip install` 安装一个，或在命令行加
`--import MODULE`。

## 2. 运行任务

```bash
uv run genelab play <task-id>
```

CLI 会在 `TASKS` 注册表中解析 `<task-id>`，构造其 `TaskCfg`，应用命令行 override，然后在
Genesis 后端跑 rollout。

三个场景短标志：

```bash
uv run genelab play <task-id> --vis           # 启用可视化
uv run genelab play <task-id> --steps 500     # 限制单 episode 步数
uv run genelab play <task-id> --gpu 1         # 锁定到单张 GPU
```

任意 override 使用 `--a.b.c VALUE` 点路径写法，字符串值按目标 dataclass 字段的类型注解自动
转换：

```bash
uv run genelab play <task-id> \
  --env.simulation.dt 0.005 \
  --env.actions.scale 0.5
```

## 3. 训练

任务自带 RL runner（`rsl_rl` 等）时，以如下方式启动训练：

```bash
uv run genelab train <task-id>
uv run genelab train <task-id> --gpus 4       # 多 GPU 通过 torchrun 启动
uv run genelab train <task-id> --checkpoint path/to/model.pt
```

!!! warning "分布式训练"
    `--gpus N` 会透明走 `torchrun` 启动分布式训练，前提是任务自身的 runner 支持 `torchrun`。

## 4. 新建下游项目

生成一个独立的下游扩展包：

```bash
uv run genelab project new my_robot_project
```

会生成 `config.py`、`robots.py`、`envs.py`、`tasks.py` 以及一份声明了 `genelab.extensions`
entry point 的 `pyproject.toml` —— 形态与仓库自带示例扩展一致。

## 5. 进阶：在 Unitree G1 上跑通完整 RL 流程 { #unitree-g1 }

仓库在 `examples/unitree/` 下提供一个生产级 RL 示例：两个 Unitree G1 人形机器人的 PPO 任务
（速度跟踪与动作模仿），从 mjlab 移植并适配到 Genesis。下面给出从安装到训练再到 checkpoint
回放的全部命令，无需跳页阅读。

### 5.1 安装扩展

unitree 扩展随包附带 MJCF 与网格资源（约 19 MB）。`torch-*` extra 需按硬件挑选 ——
对照表见 *安装*。

```bash
uv sync --extra torch-cu128
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

### 5.2 速度跟踪 PPO

速度跟踪任务要求 G1 跟随给定的 body-frame twist。训练后再回放训练好的 checkpoint：

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 1500

uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 \
    --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

`--checkpoint` 会让 `play` 走 RL runner，并把 `--agent` 默认设为 `trained`。

### 5.3 动作模仿

动作模仿任务按 body 跟踪一段录制的动作片段（BeyondMimic 风格）。默认 clip 是 LAFAN1
retargeted 的 `dance1_subject2` NPZ，首次使用时由
`genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2()` 从 `genelab-assets`
拉取并缓存到 `.cache/`，无需手动下载。

```bash
# 逐帧回放参考 clip (不接策略；机器人每步贴到 motion 当前帧)。
uv run python -m genelab_unitree.replay_motion

# 训练。
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 30000

# 回放训练好的策略。
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt
```

### 5.4 `--agent` 三种模式

`--agent` 决定 `play` 的策略来源：

| 取值 | 策略来源 |
|------|---------|
| `zero` | 恒零动作 —— 基本健康检查；机器人 reset 后在重力下倒掉。 |
| `random` | 均匀随机动作 —— 在参考姿态附近显示可见扰动。 |
| `trained` | 从 `--checkpoint` 加载。设置 `--checkpoint` 时即为默认。 |

## See also

- [配置系统](../concepts/configs.md)
- [扩展加载](../concepts/extensions.md)
- [示例](../examples/overview.md)
