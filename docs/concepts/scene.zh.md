# 场景与实体

`genelab.scene.InteractiveScene` 是组合根 (composition root)，持有一个 Genesis `gs.Scene`
以及关节机器人、刚体道具、可选地形、鼠标交互插件。`ManagerBasedRlEnv` 把所有 sim 层
编排都交给 scene；env 层只管 7 个 MDP term。

## 组合 Genesis 场景

scene 从两个传给 env 的 cfg 对象构造。`SimulationCfg` 携带 Genesis 运行时旋钮 (时间步、
并行 env 数、viewer 开关)；`InteractiveSceneCfg` 携带组合内容 (entities、terrain、sensors、
mouse 插件、BatchRenderer 开关)。scene 解析 cfg、分配 wrapper 对象，真正的 Genesis 分配
延后到 `build()` 调用。

```python
from genelab.asset_zoo import CartpoleCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

env_cfg = ManagerBasedRlEnvCfg(
    simulation=SimulationCfg(num_envs=4096, dt=0.005, substeps=1, vis=False),
    scene=InteractiveSceneCfg(env_spacing=(2.5, 2.5)),
    robot=CartpoleCfg(),
)
```

## 生命周期

`InteractiveScene` 遵循严格的三步生命周期，运行时强校验：

1. **构造**：`InteractiveScene(sim_cfg, scene_cfg, device_hint=...)` 解析 cfg 并为
   `scene_cfg.entities` 里每个实体分配 wrapper。此刻还没 import Genesis。
2. **追加实体**：`add_entity(name, cfg)` 注册额外实体——env 用它把
   `ManagerBasedRlEnvCfg.robot` 以 `"robot"` 名注入。`build()` 之后再调
   `add_entity` 会抛 `RuntimeError`。
3. **构建**：`build()` 导入 Genesis、初始化 backend、创建 `gs.Scene`、生成地面 / 地形 /
   实体 / 相机，按需挂上鼠标插件，最后 `gs_scene.build(n_envs, env_spacing)`。

`ManagerBasedRlEnv.__init__` 顺序跑完三步，build 之后再 bind 每个 articulation，让
per-joint / per-link tensor 落位。

## SimulationCfg

| 字段 | 默认 | 含义 |
|---|---|---|
| `vis` | `False` | 打开 Genesis viewer。实际只配合 `num_envs=1`；viewer 只能渲染 env 0。 |
| `gpu` | `False` | 使用 CUDA backend (`gs.gpu`)。`BatchRenderer` 需要它。 |
| `steps` | `240` | CLI `play` 在没指定 override 时默认跑的步数。 |
| `dt` | `0.01` | 物理时间步 (秒)。 |
| `substeps` | `4` | 每个物理步内的 Genesis 子步数。 |
| `num_envs` | `1` | 并行 env 数。大多数 manager 在这个维度上向量化。 |

## InteractiveSceneCfg

| 字段 | 默认 | 含义 |
|---|---|---|
| `env_spacing` | `(2.0, 2.0)` | `num_envs > 1` 时 Genesis 在 XY 上为每个 env 留的间距。 |
| `sensors` | `()` | scene 构造完后批量 build + bind 的 `SensorCfg` 元组。 |
| `mouse_interaction` | `False` | 给 viewer 挂上 GeneLab 的鼠标拖拽插件 (仅在 `vis=True` 时有效)。 |
| `entities` | `{}` | 按名字索引的额外 `ArticulationCfg` / `RigidObjectCfg`。env 会以 `"robot"` 名注入主机器人。 |
| `terrain` | `None` | `TerrainGeneratorCfg`；默认 `None` 时铺一张 `gs.morphs.Plane` 平面。 |
| `batch_render` | `False` | 把 `gs.renderers.BatchRenderer(use_rasterizer=False)` 传给 `gs.Scene`。`CameraSensor` 输出 per-env RGB-D 张量必需。仅支持 Linux x86-64 + CUDA。 |

## Articulation

`Articulation` 包装一个关节机器人。从 `ArticulationCfg` 构造，再通过
`scene.add_entity("robot", cfg)` 加入 scene (或者直接设 `ManagerBasedRlEnvCfg.robot`，让
env 自动接管)。

| 字段 | 类型 | 含义 |
|---|---|---|
| `mjcf_path` | `str` | MuJoCo XML 文件的绝对路径。 |
| `init_pos` | `tuple[float, float, float]` | 世界系初始位移 (米)。 |
| `init_quat` | `tuple[float, float, float, float]` | 世界系初始姿态 (wxyz 四元数)。 |
| `default_joint_pos` | `dict[str, float]` | 正则键控的默认关节位置；按 last-match-wins。 |
| `actuators` | `dict[str, ActuatorBaseCfg]` | 关节分组。每个被驱动关节必须恰被一个分组覆盖。 |
| `foot_link_names` | `tuple[str, ...]` | 给下游 MDP term (例如 contact sensor) 的可选元数据。 |

M3 之前的 `joint_kp` / `joint_kv` / dict 形 `action_scale` 旋钮已经移除——同等行为请
通过 `actuators` 配。被动关节用一个零增益的 `ImplicitPDActuatorCfg` 显式覆盖，保持
拓扑可见。

```python
from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg

cartpole_cfg = ArticulationCfg(
    mjcf_path="/path/to/cartpole.xml",
    init_pos=(0.0, 0.0, 1.5),
    default_joint_pos={"cart_slide": 0.0, "pole_hinge": 0.0},
    actuators={
        "cart": ImplicitPDActuatorCfg(
            target_names_expr=("cart_slide",), stiffness=80.0, damping=8.0,
        ),
        "pole": ImplicitPDActuatorCfg(
            target_names_expr=("pole_hinge",), stiffness=0.0, damping=0.0,
        ),
    },
)
```

## RigidObject

`RigidObject` 表示一个非关节刚体。无 actuator、无 per-step 状态——它就是几何，参与碰撞
与接触，但永远不被驱动。`build()` 之前通过 `scene.add_entity("name", RigidObjectCfg(...))`
加入。

| 字段 | 类型 | 含义 |
|---|---|---|
| `morph` | `Literal["plane", "box", "sphere", "mesh", "mjcf"]` | Genesis morph 选择。 |
| `file` | `str \| None` | mesh / MJCF 路径；`"mesh"` / `"mjcf"` 必填。 |
| `size` | `tuple[float, ...]` | box 是 `(x, y, z)`；sphere 是 `(radius,)`。 |
| `init_pos`、`init_quat` | tuple | 世界系初始位姿 (wxyz quat)。 |
| `fixed` | `bool` | `True` 时把刚体焊死在世界系。 |

典型用法是静态障碍、目标 marker、被动负载。只要有一个关节就用 `Articulation`，不要用
`RigidObject`。

## 已知失效模式

!!! warning "`build` 之后追加实体"
    `build()` 之后再调 `InteractiveScene.add_entity` 会抛 `RuntimeError`。先构造 env，把
    所有额外实体加完，再让 env 调用 `build()`。

!!! warning "空 `actuators` 字典"
    `Articulation.bind` 拒绝空 `actuators`。每个被驱动关节必须被恰一个分组覆盖；未匹配
    或被双重匹配的关节会带着冲突关节名抛 `ValueError`。

!!! warning "Genesis backend 不匹配"
    `SimulationCfg.gpu=True` 需要可用的 CUDA 安装。macOS / 纯 CPU Linux 上保持
    `gpu=False`；CPU backend 跑单 env play 循环没问题。`batch_render=True` 额外要求
    Linux x86-64 + CUDA + Madrona——没有 CPU 回退。

!!! note "Viewer 与并行 env"
    `SimulationCfg.vis=True` 仅在 `num_envs=1` 时有意义。viewer 只渲染 env 0，所以训练
    时把 `num_envs` 设高，再单独配一份 `num_envs=1` 的 `play_env` 用于人眼检查。

!!! tip "Rollout 中途关闭 viewer"
    用户关掉 Genesis viewer 后，内核会在 `InteractiveScene.step` 里捕获
    `GenesisException("Viewer closed.")` 并把 `env.viewer_closed` 置 `True`，后续
    `env.step` 自动 no-op。上层循环只需轮询这个 flag，不必自己写 try / except：
    ```python
    for step in range(max_steps):
        obs, *_ = env.step(action)
        if env.viewer_closed:
            break
    ```

## See also

- [Configs](configs.md)
- [Actuators](actuators.md)
- [Asset zoo](asset_zoo.md)
