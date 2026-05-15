# 性能剖析

GeneLab 在 `torch.profiler` 之上提供了一层薄壳，用于测量训练与推理阶段每一步的开销。剖析
默认关闭：`play` 与 `train` 通过 `--prof` 系列标志启用，对应的 `GENELAB_PROFILE` 环境变量
作为后备。所有跟踪结果以 TensorBoard 格式写出。

## 启用剖析器

两个命令接受同一组标志。当 CLI 标志与环境变量同时设置时，CLI 标志优先。

```bash
uv run genelab train <task-id> --prof --prof-out logs/myprof
GENELAB_PROFILE=1 GENELAB_PROFILE_OUT=logs/myprof uv run genelab train <task-id>
```

## 参数

| 标志 | 等价环境变量 | 作用 |
|------|--------------|------|
| `--prof` | `GENELAB_PROFILE=1` | 启用剖析器。 |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | 跟踪输出目录（默认 `logs/torch_profile`）。 |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | 每个周期开始前跳过的步数（默认 `10`）。 |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | 周期内的预热步数（默认 `5`）。 |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | 每周期实际记录的步数（默认 `10`）。 |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | 周期数（默认 `2`）。 |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | 记录张量形状，便于按 op 输入归因。 |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK=1` | 抓取 Python 调用栈，开销较高。 |

## 调度默认值

一个周期覆盖 `wait + warmup + active` 步，`repeat` 决定周期数。默认参数下每个周期覆盖 25
步、共采集 2 个周期，单次 `--prof` 总计记录 20 个 active 步 —— 足以暴露热点 op，又不会
生成过大的跟踪文件。

对 `train`，一步指 `RslRlVecEnvWrapper.step` 的一次 rollout 调用；对 `play`，一步指一次
策略/环境迭代。若希望以 PPO 迭代而非环境步表达调度，将 `--prof-wait`、`--prof-warmup`、
`--prof-active` 乘以 `num_steps_per_env` 即可。

## 查看跟踪结果

`genelab prof open` 会针对跟踪目录启动 TensorBoard：

```bash
uv run genelab prof open logs/torch_profile --port 6006
```

目录不存在时命令直接拒绝运行；若 `PATH` 中找不到 `tensorboard`，会给出安装提示。

## 分布式运行

!!! note "仅 rank 0 写出跟踪"
    `torchrun` 下只有 rank 0 写出跟踪文件，其余 rank 上 `maybe_profile()` 退化为 no-op，
    避免多个 worker 互相覆盖。CLI 标志会原样穿过 torchrun 的重启流程。

## See also

- [play 与 train](play-train.md)
- [配置系统](../concepts/configs.md)
- [torch.profiler 参考](https://pytorch.org/docs/stable/profiler.html)
