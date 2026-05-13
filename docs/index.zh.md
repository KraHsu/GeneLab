# GeneLab

GeneLab 是一个面向强化学习与机器人研究的 Isaac Lab 风格 API，由
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 提供仿真后端。它保留了机器人、环境、
任务注册，manager-based MDP 配置，以及 CLI 调度这些常见的组织方式。

## 目标

- 提供小型机器人、环境和任务注册表。
- 将核心 API 层与示例资产、演示脚本分离。
- 使用 manager 风格配置钩子组织 actions、observations、rewards、events 和 terminations。
- 保持 Genesis 后端集成显式，便于扩展。
- 通过稳定的包结构与 CLI 支持下游机器人研究项目。

## 快速开始

<div class="grid cards" markdown>

- :material-download:{ .lg .middle } **安装**

    ---

    准备 `uv`，挑选一个 `torch-*` extra，并完成验证。

    [:octicons-arrow-right-24: 安装](getting-started/installation.md)

- :material-rocket-launch-outline:{ .lg .middle } **运行**

    ---

    列出已注册任务并 play 一个。

    [:octicons-arrow-right-24: 快速开始](getting-started/quickstart.md)

- :material-console-line:{ .lg .middle } **CLI**

    ---

    `play`、`train`、`project new` 与 override 语法。

    [:octicons-arrow-right-24: CLI 总览](cli/overview.md)

- :material-package-variant-closed:{ .lg .middle } **扩展**

    ---

    编写下游扩展包接入注册表。

    [:octicons-arrow-right-24: 扩展加载](concepts/extensions.md)

</div>

## 要求

- Python 3.12 或更新版本。
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

## 模块速览

- `genelab.registry`：注册表、注册 helper 与扩展加载。
- `genelab.configs`：可复用的 dataclass 配置，包括 `ManagerBasedEnvCfg` 与 `TaskCfg`。
- `genelab.lab`：注册表与 manager-based 环境原语的公共 API facade。
- `genelab.envs` / `genelab.robots` / `genelab.tasks`：核心注册 helper 的命名空间。
- `genelab.actuator` / `genelab.entity` / `genelab.scene` / `genelab.sensor` /
  `genelab.terrains` / `genelab.rl`：面向机器人研究代码的扩展命名空间。

完整自动生成的模块文档见 [API 参考](api/reference.md)。
