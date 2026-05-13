# 快速开始

本节走通从 `uv sync` 到运行已注册任务的最短路径。

## 列出可用项

核心 `genelab` 包自带空注册表 —— 机器人、环境、任务都由扩展包贡献。仓库内
`examples/genelab_examples/` 通过 `pyproject.toml` 中的 `genelab.extensions` entry point
被自动发现。

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

如果某个注册表为空，请安装或导入扩展（详见 [扩展加载](../concepts/extensions.md)）。

## 运行一个任务

```bash
uv run genelab play <task-id>
```

CLI 会查找 `<task-id>` 的注册工厂，应用命令行传入的配置 override，然后在 Genesis 后端运行。

常用快捷：

```bash
uv run genelab play <task-id> --vis           # 启用可视化
uv run genelab play <task-id> --steps 500     # 限制单 episode 步数
uv run genelab play <task-id> --gpu 1         # 锁定到单张 GPU
```

任意 override 使用 `--a.b.c VALUE` 点路径写法 —— 字符串会按目标 dataclass 字段的类型注解
自动转换。完整 override 语法见 [play 与 train](../cli/play-train.md)。

## 训练（任务带 runner 时）

如果任务自带 RL runner（rsl_rl 等），可用：

```bash
uv run genelab train <task-id>
uv run genelab train <task-id> --gpus 4       # 多 GPU 通过 torchrun 启动
uv run genelab train <task-id> --checkpoint path/to/model.pt
```

`--gpus N` 会透明走 `torchrun` 启动分布式训练，前提是任务自身的 runner 支持 `torchrun`。

## 新建下游项目

当示例扩展不够用、想要自己的包时：

```bash
uv run genelab project new my_robot_project
```

生成包含 `config.py`、`robots.py`、`envs.py`、`tasks.py` 以及带 `genelab.extensions` entry
point 的 `pyproject.toml`。详见 [新建项目](../cli/project-new.md)。

## 下一步

- [CLI 总览](../cli/overview.md) —— 所有子命令与全局 flag。
- [注册表](../concepts/registry.md) —— `ROBOTS` / `ENVS` / `TASKS` 的工作机制。
- [配置系统](../concepts/configs.md) —— `ManagerBasedEnvCfg` 与 `apply_overrides`。
- [扩展加载](../concepts/extensions.md) —— 三种注册下游代码的方式。
