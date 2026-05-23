# 契约/端口

`genelab.contracts` 是 domain 代码使用的很小一层公开边界。Manager、MDP term 和 sensor
依赖的是它们需要的 env / scene **表面**，而不是当前提供这些能力的 Genesis 具体实现。

## 为什么需要这一层

大多数任务逻辑只需要少量稳定属性：device、env 数量、scene、manager、entity 表，以及写回状态的
helper。如果每个 term 都直接导入 `ManagerBasedRlEnv` 或 `InteractiveScene`，domain 层就会反向依赖
runtime 层，未来接入第二套 env 实现也会更困难。

contracts 把这个依赖反转过来：

```text
domain term -> EnvContext / SceneContext protocol
具体 env -> 实现这些 protocol
```

这是一条类型边界，不是 wrapper。运行时对象仍然是普通 env 和 scene；protocol 只记录 domain 代码可以
依赖哪些属性。

## EnvContext

`EnvContext` 是 MDP term 面向的环境表面。observation、reward、termination、event、curriculum、
metric、action 或 command 函数需要 env 时，用它做类型标注：

```python
from genelab.contracts import EnvContext


def base_height(env: EnvContext):
    robot = env.articulations["robot"]
    return robot.data.root_pos_w[:, 2]
```

这个 protocol 包含 `device`、`num_envs`、`dt`、episode length buffer、`scene`、
`articulations`、`sensors` 等只读运行时信息，也包含 domain term 会用到的 manager 属性和状态写回
helper。

## SceneContext

`SceneContext` 是通过 `env.scene` 访问到的 scene 表面。代码只需要 scene 级状态时使用它，例如
`num_envs`、`device`、`env_origins`、Genesis scene、terrain、sensor、articulation、rigid object、
recorder bridge 或 viewer 状态。

普通 MDP term 优先使用 `EnvContext`，因为它保留了 manager 和写回 helper。只有真正只依赖 scene 的
helper 才使用 `SceneContext`。

## NoiseCfg

`NoiseCfg` 现在位于 `genelab.contracts`，因为 observation manager 需要标注 noise，但不应该导入
`genelab.mdp`。`Unoise`、`Gnoise`、`ScaledNoise`、`CorrelatedNoise`、`BiasDrift` 等具体噪声模型仍位于
`genelab.mdp.noise`，并继续通过常用 MDP 与 facade 导入路径暴露。

## 继续阅读

- [模块地图](../reference/module-map.md)
- [Manager 与 MDP term](managers.md)
- [MDP term 参考](mdp.md)
