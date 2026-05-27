# 新建项目

`genelab project new` 创建独立扩展包。下游项目应使用它，而不是直接修改 `src/genelab/`。

## 用法

```bash
genelab project new my_robot_project
```

选项：

| 选项 | 说明 |
|---|---|
| `--path PATH`、`-p PATH` | 生成项目的父目录。 |
| `--package NAME` | Python 包名，默认由 `NAME` 规范化得到。 |
| `--task-id ID` | 初始 task id，默认 `<PackageName>-Example-v0`。 |
| `--force` | 目标已存在时覆盖骨架文件。 |

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

## 生成后

```bash
uv pip install -e my_robot_project
genelab list tasks
genelab play MyRobotProject-Example-v0 --steps 3
```

## 另见

- [构建扩展项目](../best-practices/extension-projects.md)
- [扩展加载](../concepts/extensions.md)
