# 扩展加载

扩展是普通 Python 包，用来向 GeneLab 注册机器人、环境和任务。真实项目应使用这种方式构建。

## 为什么扩展要分离

把下游项目放在 `src/genelab/` 外，可以避免框架变成项目专用代码集合。团队也能独立 version、install 和发布自己的机器人包。

## 发现机制

| 机制 | 适用场景 |
|---|---|
| `genelab.extensions` entry point | 已安装包和日常工作流。 |
| CLI `--import MODULE` | 临时本地模块或调试 entry-point 加载。 |
| 程序内加载 | 嵌入式应用。 |

所有机制最终做同一件事：注册函数调用 `register_robot`、`register_env`、`register_task`。

## 扩展契约

扩展应能作为包导入，保持注册阶段导入轻量，暴露无参 `register()` hook，并避免重复加载时重复注册。

## 继续阅读

- [构建扩展项目](../best-practices/extension-projects.md)
- [新建项目](../cli/project-new.md)
- [注册表](registry.md)
