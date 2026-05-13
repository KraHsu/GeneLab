# play 与 train

`play` 与 `train` 共享同一条派发路径：在 `TASKS` 注册表查找 `<task-id>`，构造其 `TaskCfg`，
应用命令行 override，然后交给单次 rollout（`play`）或任务自带的 RL runner（`train`）。

## play

```bash
uv run genelab play <task-id> [短标志] [-- 覆盖项]
```

### 短标志

这三个短标志会被改写为对应的 `env.scene.*` override：

| 标志 | 等价 override | 作用 |
|------|--------------|------|
| `--vis` | `env.scene.vis=true` | 开启 Genesis 可视化。 |
| `--gpu N` | `env.scene.gpu=N` | 把 rollout 锁定到指定 GPU。 |
| `--steps N` | `env.scene.steps=N` | 限制 episode 步数。 |

### Override 语法

短标志之后的任意 `--<dotted.path> VALUE` 都会被当作配置 override：

```bash
uv run genelab play <task-id> \
  --env.scene.dt 0.005 \
  --env.actions.scale 0.5 \
  --env.observations.include_velocity true
```

点路径会沿 `TaskCfg` 的 dataclass 树（通常以 `env.*` 为根）下钻。字符串值按目标字段类型注解
自动转换 —— 支持 `int`、`float`、`bool`、`Path`、`list[...]`、`tuple[...]`。详细的转换规则
和如何处理 union 与 `Literal` 见 [配置系统](../concepts/configs.md)。

!!! tip "短标志位置"
    CLI 内的 `_normalize_run_flags` 会把 `play --steps 5 <task-id> ...` 改写成
    `play <task-id> --steps 5 ...`，让 `argparse.REMAINDER` 正常工作。短标志放在任务 ID 前后
    都可以。

## train

```bash
uv run genelab train <task-id> [短标志] [-- 覆盖项]
```

`train` 需要任务暴露一个 RL runner（通常是 `rsl_rl_lib`）。Override 语法与 `play` 一致。额外标志：

| 标志 | 作用 |
|------|------|
| `--gpus N` | 通过 `torchrun` 启动 `N` 进程分布式训练。 |
| `--checkpoint PATH` | 从 checkpoint 文件继续训练。 |

### 多 GPU

`--gpus N` 会把底层训练入口包装成 `torchrun --standalone --nproc_per_node=N`。任务自身的 runner
必须支持 `torchrun`；依赖 Genesis 全局状态的环境通常天然兼容。

设置 `--gpus N` 时 CLI 还会按所选设备掩码 `CUDA_VISIBLE_DEVICES`，让每个 rank 看到独立 GPU。

## 示例

```bash
# 本地可视化跑一遍。
uv run genelab play wuji_hand --vis --steps 200

# 4 GPU 训练，并修改 action scale。
uv run genelab train wuji_hand --gpus 4 --env.actions.scale 0.3

# 从 checkpoint 继续。
uv run genelab train wuji_hand --checkpoint logs/wuji_hand/run_42/model_100.pt
```

## 另见

- [配置系统](../concepts/configs.md) —— `apply_overrides` 完整语义。
- [注册表](../concepts/registry.md) —— `<task-id>` 如何解析。
