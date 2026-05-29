# 如何调试常见问题

GeneLab 常见失败的最短排查路径。

## Unknown task

```bash
genelab list tasks
genelab --import my_project.tasks list tasks
```

如果第二条命令能看到任务，说明包只有在显式 `--import` 时才被导入。安装 entry point，
或继续使用显式导入。如果两条都不行，确认包已安装，或它的 `src/` 目录在 `PYTHONPATH` 上。

## Unknown override path

```bash
genelab info TASK_ID
```

从 `Overridable cfg paths` 复制路径。注意 task 有 `play_env` 时，`play` 的短标志可能把
`env.simulation.*` 重定向到 `play_env.simulation.*`。

## Torch 或 Genesis 导入错误

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import genesis; print(genesis.__version__)"
```

只选择一个 `torch-*` extra。需要时重新同步：

```bash
uv sync --reinstall-package torch --extra torch-cu128
```

## 缓存或写入目录错误

```bash
genelab cache
```

这会创建项目本地缓存目录，并把 `XDG_CACHE_HOME` 和 `MPLCONFIGDIR` 指向 `.cache/`。

## Viewer 失败

先无界面运行：

```bash
genelab play TASK_ID --steps 32
```

再尝试 viewer：

```bash
genelab play TASK_ID --vis --steps 128
```

viewer 失败通常是图形驱动、display 或 OpenGL platform 问题，而不是注册表问题。

## Observation 或 reward shape 错误

Manager term 应返回带 batch 维度的 tensor：

| Term 类型 | 期望 shape |
|---|---|
| observation term | `(num_envs, d)` 或 `(num_envs,)` |
| reward term | `(num_envs,)` |
| termination term | `(num_envs,)` bool |

避免返回标量 `()` tensor。它可能错误广播，或在 runner 中延迟失败。

## 分布式 env 数错误

`--num_envs` 表示所有 rank 的总数，必须能被 `--gpus` 整除。

```bash
genelab train TASK_ID --gpus 4 --num_envs 4096
```

想避免除法语义时使用 `--num_envs_per_gpu`：

```bash
genelab train TASK_ID --gpus 4 --num_envs_per_gpu 1024
```
