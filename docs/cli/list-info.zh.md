# 发现：list 与 info

用 `list` 查看扩展注册了什么。用 `info` 检查单个注册对象，并复制 override 路径。

## 列出注册表

```bash
genelab list robots
genelab list envs
genelab list tasks
```

如果扩展尚未通过 entry point 安装，显式导入：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

要只看随包 asset zoo（不加载用户已安装的扩展），加 `--no-entry-points`：

```bash
genelab --no-entry-points list robots
```

## 查看任意已注册名字

```bash
genelab info GeneLab-Inverted-Pendulum-v0   # task
genelab info CartPole-Env-v0                # env
genelab info anymal_c                       # robot
```

`info` 不要求事先知道是哪一类。Task 视图额外包含：

| 区域 | 含义 |
|---|---|
| 元信息 | 注册名、描述、cfg 类型、示例命令。 |
| 任务配置 | `TaskCfg` 摘要：env、robot、trainable、agent。 |
| override 路径 | `play` 和 `train` 接受的 dotted path。 |

若该 task 需要的 asset 尚未缓存，`info` 会即时下载并显示进度条
（和 `play` / `train` / `asset download` 走的是同一条 progress UI）。

## 使用复制出的 override 路径

```bash
genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

当 task 提供 `play_env` 时，play 模式下的简写标志（`--vis` / `--gpu` / `--dt` /
`--steps`）会作用到 `play_env`；显式的 dotted 路径仍按字面写的子树生效。

## 另见

- [配置参考](../reference/configuration.md)
- [CLI 参考](../reference/cli.md)
- [Asset zoo](../concepts/asset_zoo.md)
