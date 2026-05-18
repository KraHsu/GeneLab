# 配置系统

GeneLab 配置是一棵 dataclass 树。它刻意使用普通 Python，而不是自定义 schema：任务作者可以用常规代码构造配置，CLI 也仍然能暴露稳定的 override 路径。

## 核心概念

`TaskCfg` 是发现层与运行层之间的边界。它记录 task id、已注册 env 和 robot 名、默认 env 配置、
可选 play-mode env 配置、可选 agent 配置。

```text
TaskCfg
├── env
├── play_env
└── agent
```

`env` 有意标成 `object`。下游项目可以使用 `ManagerBasedRlEnvCfg` 这样的子类，而不需要修改 GeneLab core。

## 为什么 dotted override 可行

CLI 把 `--a.b.c VALUE` 当作 dataclass 树上的路径。`apply_overrides` 解析路径、读取当前值和类型注解、转换字符串值，并在运行前修改配置。

这样实验可复现：最终 `TaskCfg` 可以被 dump，非法路径会早失败，而不是被静默忽略。

## Play 配置是一等配置

训练和人工检查通常需要不同默认值。`play_env` 让 task 把 viewer 友好的设置和训练设置放在一起：
更少 env、打开 viewer、可选鼠标交互、可选 recording 输出。

## 继续阅读

- [配置参考](../reference/configuration.md)
- [设计任务](../best-practices/task-design.md)
- [CLI 参考](../reference/cli.md)
