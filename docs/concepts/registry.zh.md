# 注册表

GeneLab 使用小型注册表把框架和下游机器人项目分开。核心包不需要在导入时知道所有机器人、
环境和任务；扩展在 CLI 或用户代码需要时，把命名 factory 注册进来。

## 为什么需要注册表

机器人项目常把三件事混在一起：资产定义、环境构造、实验执行。GeneLab 把它们拆开：

| 注册表 | 边界 |
|---|---|
| `ROBOTS` | 命名机器人/资产配置 factory。 |
| `ENVS` | 命名环境 factory。 |
| `TASKS` | 命名任务 factory，持有 `TaskCfg` 与运行方法。 |

这样 CLI 可以有稳定的发现模型，而不用把所有下游项目硬编码进 `genelab`。

## 惰性 factory

注册表条目保存 factory，而不是已经构造好的对象。列名字和描述很轻量；导入 Genesis、下载资产、
构建 scene 这类昂贵工作只在 factory 被调用时发生。

因此 `genelab list tasks` 可以很快，而 `genelab play TASK` 可以启动仿真器。

## 扩展加载

扩展有三种注册方式：

| 机制 | 用途 |
|---|---|
| `genelab.extensions` entry point | 包安装后的日常使用。 |
| `--import MODULE` | 本地实验、notebook、未安装 entry point 的包。 |
| `load_extension_module()` | 程序内嵌入。 |

当模块可能在同一进程中重复加载时，注册逻辑应保持幂等。

## 继续阅读

- [构建扩展项目](../best-practices/extension-projects.md)
- [发现：list 与 info](../cli/list-info.md)
- [API 参考](../api/reference.md)
