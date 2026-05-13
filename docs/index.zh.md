# GeneLab

GeneLab 是一个面向强化学习与机器人研究的 Isaac Lab 风格 API，由
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 提供仿真后端。保留机器人、环境、
任务注册，manager-based MDP 配置，以及 CLI 调度等组织方式。

## 目标

- 小型机器人、环境和任务注册表。
- 核心 API 层与示例资产、演示脚本分离。
- 用 manager 风格配置钩子组织 actions、observations、rewards、events、terminations。
- Genesis 后端集成显式且易于扩展。
- 通过稳定的包结构与 CLI 支持下游机器人研究项目。

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

## See also

- [安装](getting-started/installation.md)
- [快速开始](getting-started/quickstart.md)
- [API 参考](api/reference.md)
