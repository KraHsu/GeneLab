# CLI 参考

GeneLab 暴露一个命令行脚本：

```bash
uv run genelab [全局选项] <命令> [命令选项]
```

## 全局选项

| 选项 | 说明 |
|---|---|
| `--version` | 打印 GeneLab 版本并退出。 |
| `--import MODULE` | 派发命令前导入扩展模块，可重复。 |
| `--no-entry-points` | 跳过已安装的 `genelab.extensions` entry point。适合做完全显式的调试。 |
| `--help` | 显示帮助。 |

## 命令

| 命令 | 用途 |
|---|---|
| `cache` | 创建项目本地缓存目录。 |
| `list robots` | 列出已注册机器人。 |
| `list envs` | 列出已注册环境。 |
| `list tasks` | 列出已注册任务。 |
| `info NAME` | 查看已注册 task、env 或 robot 的详情。 |
| `play TASK ...` | 运行已注册任务。 |
| `train TASK ...` | 用支持的 runner 训练任务。 |
| `prof open [DIR]` | 为 profiler trace 打开 TensorBoard。 |
| `project new NAME` | 生成外部 GeneLab 扩展项目骨架。 |

## 运行时标志

这些标志放在 `play` 或 `train` 的 task id 后面。

| 标志 | 说明 |
|---|---|
| `--vis`、`-v` | 启用 Genesis viewer。 |
| `--headless` | 强制无 viewer（`env.simulation.vis=false`），与 `--vis` 互斥。在无显示器的服务器上跑 `play --agent trained` 时需要（其 play env 默认会启用 viewer）。 |
| `--gpu` | 使用 Genesis GPU 后端。 |
| `--steps N` | 在 `play` 中表示 rollout 步数；在 `train` 中是 `--max_iterations N` 的短写。 |
| `--dt X` | 覆盖仿真时间步。 |
| `--a.b.c VALUE` | 应用 dotted config override。 |

## Runner 标志

| 标志 | 命令 | 说明 |
|---|---|---|
| `--num_envs N` | play/train | env 总数。分布式 train 中会除以 `--gpus`。 |
| `--num_envs_per_gpu N` | play/train | 每个 rank 的 env 数。与 `--num_envs` 互斥。 |
| `--agent zero|random|trained` | play | 选择策略来源。 |
| `--checkpoint PATH` | play/train | 加载 checkpoint。play 中会默认把 agent 设为 `trained`。 |
| `--seed N` | train | 覆盖 env 和 agent seed。 |
| `--log_dir PATH` | train | 使用已解析日志目录。 |
| `--max_iterations N` | train | 覆盖 PPO iteration 数。 |
| `--gpus N` | train | 用 `torchrun` 重新启动为 `N` 个本地 rank。 |

## Profiler 标志

| 标志 | 环境变量兜底 | 默认 |
|---|---|---|
| `--prof` | `GENELAB_PROFILE=1` | 关闭 |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | `logs/torch_profile` |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | `10` |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | `5` |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | `10` |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | `2` |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | 关闭 |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK=1` | 关闭 |

## 补全与交互式恢复

Typer 提供 `--install-completion` 和 `--show-completion`。补全会加载已安装 entry point，
但看不到临时 `--import MODULE` 扩展。

stdin 是 TTY 时，GeneLab 可对缺失 task id、未知名字、非法 `--agent`、未知 override 路径
弹出选择器。脚本和 CI 中会直接抛出错误。
