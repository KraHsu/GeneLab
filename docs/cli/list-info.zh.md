# 发现：list 与 info

用 `list` 查看扩展注册了什么。用 `info` 检查单个注册对象，并复制 override 路径。

## 列出注册表

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

如果扩展尚未安装，显式导入：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 查看任务

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

任务视图包括：

| 区域 | 含义 |
|---|---|
| 元信息 | 注册名、描述、cfg 类型、示例命令。 |
| 任务配置 | `TaskCfg` 摘要：env、robot、trainable、agent。 |
| 可覆盖路径 | `play` 和 `train` 接受的 dotted path。 |

## 使用复制出的 override 路径

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

当 task 有 `play_env` 时，play 模式短标志会作用到 `play_env`；显式路径仍保持显式语义。

## 另见

- [配置参考](../reference/configuration.md)
- [CLI 参考](../reference/cli.md)
