# 执行器

`genelab.actuator` 是 manager-based RL 策略与 Genesis 之间的机电层。每个 `ActuatorBase` 通过正则表达式占有一段 articulation 的可驱动关节，并决定单步关节位置目标如何抵达仿真器：要么走 Genesis 的隐式 PD 通道，要么在 Python 端算出力矩并通过 `control_dofs_force` 推下去。

## 为什么需要专门的一层

真实机器人不是单一规格 —— 一台 Unitree G1 同时挂着 6 种不同电机，分布在膝、髋、肩、踝。各类电机的 stiffness、damping、effort、velocity、armature、friction 都不一样。把这些信息全压进 `ArticulationCfg` 上扁平的 `joint_kp` / `joint_kv` 字典会丢掉电机身份，也无法表达速度相关的力矩饱和。actuator 命名空间把分组抽象重新拿回来，并加上 3 种力矩模型。

## 自带 3 种模型

| 类 | 通道 | 力矩计算 |
|---|---|---|
| `ImplicitPDActuator` | `implicit_pd` | 不算 —— Genesis 内部求解 `tau = kp*(q* - q) - kv*q_dot` |
| `IdealPDActuator` | `force` | `tau = clip(kp*(q* - q) - kv*q_dot, ±effort_limit)` |
| `DCMotorActuator` | `force` | `IdealPD` 加上仅驱动方向生效的线性退磁 |

`ImplicitPDActuator` 与 M2 之前的实现数值等价 —— 对 Genesis 调 `set_dofs_kp` / `set_dofs_kv`，每步目标通过 `control_dofs_position` 进入仿真。`IdealPDActuator` 把仿真器内部 PD 增益置零，成为带显式 effort 上限的力矩控制规范路径。`DCMotorActuator` 在上面追加一条力矩-速度退磁曲线：

`tau_max(q_dot) = saturation_effort * clip(1 - |q_dot| / velocity_limit, 0, 1)`

退磁仅作用于驱动方向（`tau_pd` 与 `q_dot` 同号时）；反向制动力矩保持完整的 `effort_limit`。这是 Isaac Lab `DCMotor` 的语义 —— 反电动势饱和不惩罚回馈制动。

## 装配到 articulation 上

`ArticulationCfg.actuators` 是 `dict[str, ActuatorBaseCfg]`。每一个可驱动关节都必须被恰好一个组覆盖；未匹配和正则冲突都会在 `Articulation.bind` 时抛 `ValueError`。被动关节也要写一条零增益 `ImplicitPDActuatorCfg`，让拓扑结构在 config 中可见。

```python
from genelab.actuator import DCMotorActuatorCfg, ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg

cfg = ArticulationCfg(
    mjcf_path="/path/to/robot.xml",
    default_joint_pos={"cart_slide": 0.0, "pole_hinge": 0.0},
    actuators={
        "cart": ImplicitPDActuatorCfg(
            target_names_expr=("cart_slide",), stiffness=80.0, damping=8.0,
            action_scale=1.0,
        ),
        "pole": ImplicitPDActuatorCfg(
            target_names_expr=("pole_hinge",), stiffness=0.0, damping=0.0,
        ),
    },
)
```

`target_names_expr` 是一组正则模式，对 articulation 的关节名做匹配。每组的 `action_scale` 通过 `Articulation.action_scale_tensor` 暴露 —— 当 `JointPositionAction.scale` 留 `None` 时（推荐默认）就会从这里取值。

## 需要知道的失败模式

* **`actuators` 字典为空** —— articulation 拒绝完全没有声明 actuator 的配置。哪怕是被动单摆，也要为它的 hinge 写一条零增益组。
* **未匹配关节** —— `ValueError` 列出未被任何正则覆盖的关节。
* **冲突分组** —— `ValueError` 列出被多个组同时命中的关节及涉及的组名。
* **`DCMotorActuatorCfg.velocity_limit` 为 `None`** —— 退磁曲线无断点；`__post_init__` 在 actuator 实际运行前先抛错。

## 在 G1 上切换模型

仓库自带的 `examples/unitree` 扩展声明了 6 个 `DCMotorActuatorCfg` 组（5020、7520_14、7520_22、4010、waist、ankle）。如果想让同一台机器人走仿真器隐式 PD，把每条 `DCMotorActuatorCfg(...)` 替换为同样 `stiffness` / `damping` / `effort_limit` 的 `ImplicitPDActuatorCfg(...)` 即可。articulation 不对哪个通道被激活做全局假设 —— 每个组各自决定。

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
