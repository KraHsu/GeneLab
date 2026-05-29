# 项目：project new

`genelab project new` 创建独立扩展包。下游项目应使用它，而不是直接修改 `src/genelab/`。
`project` 是 Typer 子命令组：`genelab project`（不带子命令）会列出可用动作。

## 用法

```bash
genelab project new my_robot_project
```

选项：

| 选项 | 说明 |
|---|---|
| `--path PATH`、`-p PATH` | 生成项目的父目录（默认 `.`）。 |
| `--package NAME` | Python 包名，默认由 `NAME` 规范化得到。 |
| `--task-id ID` | 初始 task id，默认 `<PackageName>-Example-v0`。 |
| `--force` | 目标已存在且非空时覆盖骨架文件；目标存在但非目录时仍会报错。 |

## 生成结构

```text
my_robot_project/
├── README.md
├── pyproject.toml
└── src/my_robot_project/
    ├── __init__.py
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

生成的 `pyproject.toml` 通过相对路径 (`tool.uv.sources`) 引用本地 GeneLab 源码，
以便不必发布也能 `uv pip install -e .`。

## 生成后

```bash
uv pip install -e my_robot_project
genelab list tasks
genelab play MyRobotProject-Example-v0 --steps 3
```

如果目标目录非空且未传 `--force`，命令会以非零状态退出而不会覆盖现有文件。

## 另见

- [构建扩展项目](../best-practices/extension-projects.md)
- [扩展加载](../concepts/extensions.md)
