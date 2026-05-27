# 如何构建扩展项目

大部分情况下，机器人、任务或实验都不应该直接放进 `src/genelab/`，应使用扩展项目。这是 `GeneLab` 下游研究代码的推荐形态。

## 1. 生成骨架

```bash
uv run genelab project new my_robot_project
```

生成的包结构：

```text
my_robot_project/
├── pyproject.toml
└── src/my_robot_project/
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

## 2. 分清文件职责

| 文件 | 职责 |
|---|---|
| `config.py` | 项目自己的 dataclass。 |
| `robots.py` | 机器人资产/配置 factory。 |
| `envs.py` | 运行时 env 类或 env factory helper。 |
| `tasks.py` | `register()` hook 和 task 类。 |

除非对象确实需要 Genesis，否则不要在模块导入时启动 Genesis。注册表发现应保持轻量。

## 3. 通过单个 hook 注册

暴露无参 `register()` 函数：

```python
def register() -> None:
    register_robot(...)
    register_env(...)
    register_task(...)
```

hook 可能运行多次时，加幂等保护：

```python
if "my-robot" not in ROBOTS:
    register_robot(...)
```

## 4. 日常使用优先 entry point

在 `pyproject.toml` 中：

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project.tasks:register"
```

然后 editable 安装：

```bash
uv pip install -e my_robot_project
uv run genelab list tasks
```

临时实验使用显式导入：

```bash
PYTHONPATH=my_robot_project/src \
  uv run genelab --import my_robot_project.tasks list tasks
```

## 5. 验证包边界

从包目录外运行：

```bash
uv run python -c "import my_robot_project"
uv run genelab list tasks
uv run genelab info MyProject-Example-v0
uv run genelab play MyProject-Example-v0 --steps 3
```

## 预期结果

项目能作为普通 Python 包导入，GeneLab 能发现 entry point，所有 task 命令都能运行，且不需要修改 GeneLab core。
