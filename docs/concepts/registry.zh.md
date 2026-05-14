# 注册表

GeneLab 提供泛型 `Registry[T]` 与三个由 `genelab.lab` 导出的模块级单例：

| 单例 | 持有内容 |
|------|---------|
| `ROBOTS` | 机器人工厂。 |
| `ENVS` | 环境工厂。 |
| `TASKS` | 任务工厂（每个任务把环境与可选 runner / agent 配置打包）。 |

## 条目结构

注册条目是一个四元组：**`(name, description, factory, cfg_type)`**。`factory` 是一个可调用，
只在 `get(name)` 时才会被调用，因此导入注册位点不会立即构造重型对象。

## 接口速览

```python
from genelab.lab import ROBOTS, ENVS, TASKS, TaskCfg

def make_my_robot():
    ...

ROBOTS.register(
    name="my_robot",
    description="A demo robot.",
    factory=make_my_robot,
    cfg_type=None,
)

# 之后：
robot = ROBOTS.get("my_robot")
```

`name` 是规范查找键；`description` 出现在 `genelab list robots` 输出里。

## 幂等扩展加载

注册模块维护 `_loaded_extension_modules` 与 `_loaded_entrypoints` 两个集合，重复加载同一
扩展会被短路。entry-point 自动发现（默认）与显式 `--import` 同时启用时，这一点尤其
重要 —— 两条路径最终都会调用 `load_extension_module`，但第二次调用直接返回。

## See also

- [扩展加载](extensions.md)
- [API 参考](../api/reference.md)
