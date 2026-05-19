# 性能剖析

GeneLab 通过 `genelab.rl.maybe_profile` 把 profiler 标志转发给 `torch.profiler`。

## 启用 profiling

```bash
uv run genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
```

打开 trace：

```bash
uv run genelab prof open logs/torch_profile
```

## 参数

| 标志 | 含义 |
|---|---|
| `--prof` | 启用 profiling。 |
| `--prof-out PATH` | trace 输出目录。 |
| `--prof-wait N` | 初始等待步数。 |
| `--prof-warmup N` | 记录前 warmup 步数。 |
| `--prof-active N` | 每个周期记录的步数。 |
| `--prof-repeat N` | 周期数。 |
| `--prof-record-shapes` | 记录 tensor shape。 |
| `--prof-with-stack` | 捕获 Python stack，开销更高。 |

## 实用默认值

先跑短 profile。GeneLab 的 RL loop 每个 env step 推进一次 profiler step，大规模 vectorized run 的 trace 会增长很快。

## 分布式运行

只有 main process 写 profiler trace。profile 分布式训练时保持 `--prof-active` 较小。

## 另见

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [CLI 参考](../reference/cli.md)
