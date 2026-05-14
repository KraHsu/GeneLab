# play 与 train

`play` 与 `train` 共享同一条派发路径：在 `TASKS` 注册表查找 `<task-id>`，构造其 `TaskCfg`，
应用命令行 override，然后交给单次 rollout（`play`）或任务自带的 RL runner（`train`）。

## play

```bash
uv run genelab play <task-id> [短标志] [-- 覆盖项]
```

短标志可放在 `<task-id>` 前或后；CLI 内部会自动整理顺序。

### 仿真短标志

下列短标志会被改写为对应的 `env.simulation.*` override：

| 标志 | 等价 override | 作用 |
|------|--------------|------|
| `--vis` | `env.simulation.vis=true` | 开启 Genesis 可视化。 |
| `--gpu N` | `env.simulation.gpu=N` | 把 rollout 锁定到指定 GPU。 |
| `--steps N` | `env.simulation.steps=N` | 限制 episode 步数。 |

### 运行时标志

任务携带 RL agent 配置时，`play` 还接受下列 runner 侧标志：

| 标志 | 作用 |
|------|------|
| `--checkpoint PATH` | 加载已训练策略并做推理 rollout；会让 `--agent` 默认为 `trained`。 |
| `--num-envs N` | 覆盖任务注册时的 env 数量（并行环境数）。 |
| `--agent {zero,random,trained}` | 选择策略来源。设置 `--checkpoint` 时默认 `trained`，否则默认 `zero`。 |

### Override 语法

短标志之后的任意 `--<dotted.path> VALUE` 都会被当作配置 override：

```bash
uv run genelab play <task-id> \
  --env.simulation.dt 0.005 \
  --env.actions.scale 0.5 \
  --env.observations.include_velocity true
```

点路径沿 `TaskCfg` 的 dataclass 树（通常以 `env.*` 为根）下钻。字符串值按目标字段类型注解
自动转换 —— 支持 `int`、`float`、`bool`、`Path`、`list[...]`、`tuple[...]`。

## train

```bash
uv run genelab train <task-id> [短标志] [-- 覆盖项]
```

`train` 需要任务暴露一个 RL runner（通常是 `rsl_rl_lib`）。Override 语法与 `play` 一致。

| 标志 | 作用 |
|------|------|
| `--gpus N` | 通过 `torchrun --standalone --nproc_per_node=N` 启动分布式训练。任务自身的 runner 必须支持 `torchrun`。 |
| `--checkpoint PATH` | 从 checkpoint 文件继续训练。 |
| `--num-envs N` | 覆盖任务注册时的 env 数量（并行环境数）。 |
| `--max-iterations N` | 限制 PPO 学习迭代次数。 |
| `--seed N` | 覆盖 runner 使用的随机种子。 |
| `--log-dir PATH` | 覆盖日志根目录，默认 `logs/<runner>/<experiment>/<run>`。 |
| `--agent {zero,random,trained}` | 转发给 runner；多用于诊断式运行。 |

当 `--gpus N > 1` 时，CLI 还会按所选设备掩码 `CUDA_VISIBLE_DEVICES`，让每个 rank 看到独立
GPU。

## 示例

```bash
# 本地可视化跑一遍。
uv run genelab play wuji_hand --vis --steps 200

# 4 GPU 训练，并修改 action scale。
uv run genelab train wuji_hand --gpus 4 --env.actions.scale 0.3

# 从 checkpoint 继续。
uv run genelab train wuji_hand --checkpoint logs/wuji_hand/run_42/model_100.pt
```

## See also

- [配置系统](../concepts/configs.md)
- [注册表](../concepts/registry.md)
