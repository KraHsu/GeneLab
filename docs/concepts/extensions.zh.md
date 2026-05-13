# 扩展加载

GeneLab 核心**不**自带机器人、环境、任务，所有内容都由下游扩展包提供。CLI 通过三条路径发现
扩展，按以下优先级：

## 1. Entry-point 自动发现（推荐）

在扩展的 `pyproject.toml` 中声明：

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project:register"
```

CLI 启动时自动 import `genelab.extensions` 组下的每个 entry point。`register()` 是一个无参
可调用，位于扩展包顶层，负责真正调用 `ROBOTS.register(...)` / `ENVS.register(...)` /
`TASKS.register(...)`（或仅 import 一些以副作用方式注册的模块）。

这条路径正是 `genelab project new` 默认替你接好的。

## 2. 显式 `--import`

```bash
uv run genelab --import my_pkg.module1 --import my_pkg.module2 list tasks
```

可重复。适用场景：

- 扩展尚未提供 `genelab.extensions` entry point。
- 想要完全显式、可复现的加载顺序（搭配 `--no-entry-points`）。
- 调试尚未安装的包，需要把源码目录临时塞进 `sys.path`（CLI 也会把当前工作目录加入 `sys.path`）。

## 3. 程序内调用

```python
from genelab.registry import load_extension_module

load_extension_module("my_pkg.module")
```

测试与嵌入脚本使用。同样的幂等守卫生效，因此即使 CLI 已经自动发现过同一模块，再调用一次
也是 no-op。

## 关闭自动发现

```bash
uv run genelab --no-entry-points --import my_pkg list tasks
```

这是最可复现的组合 —— 只加载你显式命名的模块。

## 参考形态

`examples/genelab_examples/` 是参考形态：`pyproject.toml` 带 entry point、顶层有 `register()`
可调用、并提供 `config.py` / `robots.py` / `envs.py` / `tasks.py`。`tests/fake_extension.py`
是同样形态但削减到最小，供测试套件使用。

## 另见

- [新建项目](../cli/project-new.md) —— 生成新的扩展包骨架。
- [注册表](registry.md) —— `register()` 实际做了什么。
