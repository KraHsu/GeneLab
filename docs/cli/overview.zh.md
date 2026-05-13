# CLI 总览

`genelab` 通过 `genelab = "genelab.cli:main"` entry point 暴露为命令行脚本。配合 `uv` 时
通常这样调用：

```bash
uv run genelab [全局选项] <子命令> [参数]
```

## 子命令

| 子命令 | 作用 |
|--------|------|
| `cache` | 创建项目本地仿真缓存目录（`.cache/`），并设置 `XDG_CACHE_HOME` / `MPLCONFIGDIR`。 |
| `list robots` | 列出 `ROBOTS` 注册表里已注册的机器人。 |
| `list envs` | 列出 `ENVS` 注册表里已注册的环境。 |
| `list tasks` | 列出 `TASKS` 注册表里已注册的任务。 |
| `play` | 在 Genesis 后端运行已注册任务，支持配置 override。 |
| `train` | 任务带 RL runner 时启动训练，通过 `torchrun` 支持多 GPU。 |
| `project new` | 生成一个新的下游扩展包骨架，三种注册表全部接通。 |

## 全局选项

下列标志放在任意子命令之前：

| 标志 | 作用 |
|------|------|
| `--version` | 打印 GeneLab 版本并退出。 |
| `--import MODULE` | 在派发子命令前显式导入扩展模块。可重复。适合扩展尚未提供 `genelab.extensions` entry point 时使用。 |
| `--no-entry-points` | 跳过通过 `genelab.extensions` entry-point 组的自动发现。与 `--import` 搭配可实现完全显式、可复现的扩展加载。 |

## 扩展发现顺序

CLI 启动时按三条路径依次发现扩展：entry-point 自动发现、显式 `--import MODULE`、程序内的
`genelab.registry.load_extension_module(...)`。

## See also

- [play 与 train](play-train.md)
- [新建项目](project-new.md)
- [扩展加载](../concepts/extensions.md)
