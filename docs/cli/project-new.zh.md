# 新建项目

`genelab project new` 生成一个下游扩展包骨架 —— 一个独立 Python 项目，把机器人、环境、
任务注册进 GeneLab 的全局注册表。

## 用法

```bash
uv run genelab project new my_robot_project
```

### 选项

| 选项 | 默认 | 作用 |
|------|------|------|
| `--path PATH` | `./<name>` | 输出目录，不存在时自动创建。 |
| `--package NAME` | 由项目名推导 | Python 导入名（如 `my_robot_project`）。 |
| `--task-id ID` | `<package>/<name>-v0` | 骨架注册的首个任务 ID。 |
| `--force` | 关闭 | 已存在目标目录时覆盖。慎用。 |

## 生成的目录结构

```
my_robot_project/
├── pyproject.toml        # 含 [project.entry-points."genelab.extensions"]
├── README.md
└── src/
    └── my_robot_project/
        ├── __init__.py   # 暴露 register() entry-point 调用入口
        ├── config.py     # 接入 TaskCfg.env 的任务专属 dataclass
        ├── robots.py     # 机器人注册
        ├── envs.py       # 环境注册
        └── tasks.py      # 任务注册（使用 --task-id）
```

生成的 `pyproject.toml` 声明：

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project:register"
```

安装后（`uv pip install -e ./my_robot_project`），扩展在下次启动 CLI 时被自动发现，无需任何
`--import` 标志。

## 后续步骤

```bash
cd my_robot_project
uv pip install -e .          # 安装扩展到 GeneLab 的 venv
uv run genelab list tasks    # 确认新任务 ID 出现
uv run genelab play <task-id> --vis
```

## See also

- [扩展加载](../concepts/extensions.md)
- [配置系统](../concepts/configs.md)
