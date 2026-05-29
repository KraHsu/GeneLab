# 场景与实体

`InteractiveScene` 是 GeneLab 的 Genesis scene owner。它把声明式配置转换成 live Genesis handle，
并向环境其他部分暴露 Isaac Lab 风格的 entity wrapper。

## 场景边界

`InteractiveSceneCfg` 描述应该存在什么：env spacing、entities、terrain、sensors、recordings、
viewer interaction、batch rendering。`InteractiveScene` 持有实际存在的东西：Genesis `Scene`、
articulation、rigid object、sensor、terrain importer、recorder bridge 和 viewer 状态。

这种分离让配置可序列化，也让任务能在 Genesis 启动前被检查。

## 实体

| Entity | 用途 |
|---|---|
| `Articulation` | 机器人 wrapper，包含 joint/link 名、默认 joint state、limits、刷新后的 `RobotState`。 |
| `RigidObject` | 非关节物体 wrapper。 |
| `RobotState` | observation、reward、sensor、event 读取的 batched tensor。 |

`ManagerBasedRlEnv` 把配置中的机器人作为 `"robot"` articulation 加入。实时机器人数据通过具名
entity 表访问，例如用 `env.articulations["robot"].data` 读取 `RobotState`，用
`env.articulations["robot"].joint_names` 读取关节元数据。环境也暴露 `env.scene` 供 scene 级访问。

## 机器人描述格式

`ArticulationCfg` 接受 MJCF 或 URDF 两种来源：

- `mjcf_path` —— MuJoCo `.xml` 的绝对路径。通过 `gs.morphs.MJCF` 加载。
- `urdf_path` —— `.urdf`、`.xacro` 或 `.urdf.xacro` 的绝对路径。通过
  `gs.morphs.URDF` 加载，xacro 文件由 Genesis 在加载时调用 `xacro` Python 包预处理。
  用 `xacro_args={"key": "value", ...}` 覆盖 `<xacro:arg>` 声明。

两者必须只设一个；都设时 `Articulation` 构造抛 `ValueError`。`xacro_args` 只
对 `urdf_path` 有效。

## 为什么需要 wrapper

Genesis API 和 Isaac Lab 风格任务代码使用的词汇不同。wrapper 隔离这种差异：MDP term 读取稳定的 GeneLab 属性，后端集成负责 Genesis 细节。

## Avatar（运动学）实体

`Avatar` 实体用于：场景里需要出现一个可见物体，但**不**参与物理模拟 ——
典型用法是 motion clip 鬼影、目标参考标记，或"跟随领导者"演示者，env 每步
写一次 pose 但不对场景其它物体产生力。

```python
from genelab.entity import Avatar, AvatarCfg

ghost = Avatar(
    AvatarCfg(morph="box", size=(0.1, 0.1, 0.1), init_pos=(0.0, 0.0, 1.0)),
    name="motion_ghost",
)
ghost.spawn(scene)
# 每步：
ghost.set_pose(target_pos, target_quat)
```

底层用 Genesis 1.0 的 `Kinematic` 材质，刚体求解器不管它（无接触、无约束），
但传感器和相机依然能看到。

## 批渲染的光栅化开关

`InteractiveSceneCfg.use_rasterizer`（默认 `False`）控制 `batch_render=True`
时 `BatchRenderer` 用什么后端：

- `False` —— raytracer（默认；保真度高、较慢）。
- `True` —— rasterizer（不需要照片级真实时，批渲染显著更快）。

`batch_render=False` 时忽略此项，因为没有构造 `BatchRenderer`。

## 相机调试辅助

`InteractiveScene` 提供两个对 Genesis 1.0 debug-draw API 的薄封装，用于检查
挂载在场景上的相机：

- `draw_camera_frustums(camera_names=None, color=(1, 1, 1, 0.3))` 绘制场景上
  每一个（或指定名字的）`CameraSensor` 的视锥。返回实际绘制的视锥数。指定
  的名字若没有对应 `CameraSensor` 会抛错。
- `draw_camera_trajectory(positions, radius=0.002, color=(1, 0.5, 0, 0.8))`
  在 `positions`（每行一个世界系坐标点）上画一条折线。调用方持有 buffer —— 
  通常是录制下来的相机路径 —— 所以 scene 不维护任何历史。

两个辅助方法都必须在 `InteractiveScene.build()` 之后调用；build 之前调用会抛
`RuntimeError`。一般是从 play 阶段的 hook 里调用，用来可视化 RGB-D 传感器
所看的视野以及一次 rollout 中位姿的变化。

## 继续阅读

- [模块地图](../reference/module-map.md)
- [传感器](sensors.md)
- [执行器](actuators.md)
