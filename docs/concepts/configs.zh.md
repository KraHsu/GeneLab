# 配置系统

GeneLab 的配置体系是 `genelab.configs` 下的一个小型 dataclass 层级：

```
TaskCfg
└── env: object   # 下游扩展把自己的 env dataclass 接入这里
    ManagerBasedEnvCfg
    ├── simulation
    ├── scene
    ├── actions
    ├── observations
    ├── rewards
    ├── terminations
    └── events
```

`TaskCfg.env` 类型刻意写成 `object` —— 下游扩展无需触碰核心即可接入自己的 env dataclass。
大多数情况下 `env` 字段会是 `ManagerBasedEnvCfg` 的子类。

## apply_overrides

核心函数是 `apply_overrides(cfg, dict)`，解析点路径并把值应用到配置树：

```python
from genelab.configs import apply_overrides

apply_overrides(cfg, {
    "env.simulation.dt": "0.005",
    "env.simulation.steps": "500",
    "env.actions.scale": "0.3",
    "env.observations.include_velocity": "true",
})
```

### 类型转换

字符串值按目标 dataclass 字段的类型注解自动转换：

| 类型注解 | 接受的字符串 |
|---------|-------------|
| `bool` | `true`/`false`/`1`/`0`/`yes`/`no`（大小写不敏感） |
| `int` | 任意 10 进制整数字面量 |
| `float` | 任意 Python float 字面量 |
| `Path` | 经过 `pathlib.Path(...)` |
| `list[T]` | 逗号分隔，每个元素按 `T` 转换 |
| `tuple[T, ...]` | 逗号分隔，每个元素按 `T` 转换 |
| `str` | 原样 |

当点路径无法解析到已知字段，或值无法转换时，`apply_overrides` 在配置构造期就抛出明确错误，
避免错误延后到仿真运行时才暴露。

### CLI 转发路径

`play` / `train` 把每个 `--<a.b.c> VALUE` 标志转发给 `apply_overrides`。三个仿真短标志
（`--vis`、`--gpu`、`--steps`）在转发前被改写为 `env.simulation.{vis,gpu,steps}`。

## See also

- [play 与 train](../cli/play-train.md)
- [API 参考](../api/reference.md)
