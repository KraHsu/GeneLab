# 性能剖析

GeneLab 通过 `genelab.rl.maybe_profile` 把 profiler 标志转发给 `torch.profiler`。

## 启用性能剖析

```bash
genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
```

打开 trace：

```bash
genelab prof open                       # 默认 logs/torch_profile
genelab prof open <log_dir>             # 指定目录
genelab prof open <log_dir> --port 6007 # 指定 TensorBoard 端口
genelab prof open <log_dir> --host 0.0.0.0
```

## 标志

| 标志 | 等价环境变量 | 含义（默认） |
|---|---|---|
| `--prof` | `GENELAB_PROFILE` | 启用 profiling。 |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | TensorBoard trace 目录。 |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | 初始等待步数（10）。 |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | 记录前 warmup 步数（5）。 |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | 每个周期记录的步数（10）。 |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | 周期数（2）。 |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES` | 记录 tensor shape。 |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK` | 捕获 Python stack，开销更高。 |

命令行标志始终优先于环境变量。

## 实用默认值

先跑短 profile。GeneLab 的 RL loop 每个 env step 推进一次 profiler step，大规模
vectorized run 的 trace 会增长很快。

## 分布式运行

只有 main process 写 profiler trace。profile 分布式训练时保持 `--prof-active` 较小。

## `prof open` 前置条件

- `tensorboard` 必须在 PATH 上（`uv pip install tensorboard`，或本仓库的 `rl` extra）。
- 指定的目录必须存在；空目录是合法的。

## 另见

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [CLI 参考](../reference/cli.md)
