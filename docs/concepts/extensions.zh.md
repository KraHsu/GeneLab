# 扩展加载

扩展是普通 Python 包，用来向 GeneLab 注册**机器人、环境、任务和 RL 后端**。真实项目应使用这种方式构建。

## 为什么扩展要分离

把下游项目放在 `src/genelab/` 外，可以避免框架变成项目专用代码集合。团队也能独立 version、install 和发布自己的机器人包。

## `genelab.extensions` API

`genelab.extensions` 是扩展所需一切的唯一稳定导入路径。底层实现位于 `genelab.registry`（机器人 / 环境 / 任务）和
`genelab.rl.backends`（后端），但 `genelab.extensions` 是规范入口：

```python
from genelab.extensions import (
    register_robot, register_env, register_task, register_backend,
    ROBOTS, ENVS, TASKS,        # 三个注册表本身
    Runnable, Backend,          # 两个 Protocol 契约
)
```

| 扩展类型 | 注册函数 | 存放于 |
|---|---|---|
| 机器人 cfg | `register_robot(name, factory, *, description, …)` | `ROBOTS` |
| 环境 cfg | `register_env(name, factory, *, description, …)` | `ENVS` |
| 任务 | `register_task(name, factory, *, description, …)` | `TASKS` |
| RL 后端 | `register_backend(backend)` | 按 `backend.cfg_type` 索引 |

机器人 / 环境 / 任务的 `register_*` 接收稳定的 `name` 和一个零参 `factory`，按需构建 config（注册表保持惰性）。
`register_backend` 接收一个构建完毕的后端对象，并按它所拥有的 agent-config 类型来索引。

## 发现机制

| 机制 | 适用场景 |
|---|---|
| `genelab.extensions` 分组的 entry point | 已安装包和日常工作流。 |
| CLI `--import MODULE` | 临时本地模块或调试 entry-point 加载。 |
| 程序内加载 | 嵌入式应用。 |

所有机制最终做同一件事：注册函数调用 `register_robot`、`register_env`、`register_task` 或 `register_backend`。

## `Runnable` 任务契约

任务工厂产出的每个值都必须满足 `genelab.extensions.Runnable` —— 这是 CLI 的 `play` / `train` 命令所依赖的契约：

```python
class Runnable(Protocol):
    cfg: object
    def play(self) -> None: ...
    def train(self) -> None: ...
```

因此任务类型通过 `.cfg` 暴露其配置，并实现 `play()` / `train()`。内置任务类型已满足该契约；第三方任务类型应实现相同形状。

## `Backend` 契约

一个后端拥有一个 RL 库，按任务 agent config 的**类型**被选中（见 [RL 运行器](rl-runner.md)）。契约是
`genelab.extensions.Backend`：

```python
@runtime_checkable
class Backend(Protocol):
    name: str
    cfg_type: type
    def train(self, ctx: TrainContext) -> Path: ...
    def play(self, ctx: PlayContext) -> None: ...
    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup: ...
```

`TrainContext` / `PlayContext`（位于 `genelab.rl.backends`）打包了已解析好的 env、各配置、seed、`log_dir`
以及 profiler 选项，因此库相关代码全部留在后端内部。

一个会自我注册并记录一次运行的最小后端如下：

```python
import json
from dataclasses import dataclass
from pathlib import Path

from genelab.extensions import register_backend
from genelab.rl.backends import PlayContext, TrainContext
from genelab.rl.backends.base import InferenceSetup
from genelab.rl.config import BackendConfig


@dataclass
class EchoAgentCfg(BackendConfig):
    """该后端拥有的 agent-config 类型；任务把 ``cfg.agent`` 设为它的实例。"""

    max_iterations: int = 10


class EchoBackend:
    name = "echo"
    cfg_type = EchoAgentCfg

    def train(self, ctx: TrainContext) -> Path:
        log_dir = ctx.log_dir or Path("logs") / ctx.task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run.json").write_text(
            json.dumps({"task": ctx.task_id, "iterations": ctx.max_iterations})
        )
        return log_dir

    def play(self, ctx: PlayContext) -> None:
        ctx.env.reset()

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        raise NotImplementedError("EchoBackend 不支持 eval / export")


def register() -> None:
    register_backend(EchoBackend())
```

把 `register` 暴露为包的 entry point（或通过 `--import` 调用），任何 `cfg.agent` 为 `EchoAgentCfg` 的任务都会
路由到 `EchoBackend`。`make_inference_setup` 仅在 `genelab eval` / `genelab export` 时需要；只做训练的后端可以
不实现它。

## 扩展契约

扩展应能作为包导入，保持注册阶段导入轻量，暴露无参 `register()` hook，并避免重复加载时重复注册。

## 继续阅读

- [构建扩展项目](../best-practices/extension-projects.md)
- [新建项目](../cli/project-new.md)
- [注册表](registry.md)
- [RL 运行器](rl-runner.md)
