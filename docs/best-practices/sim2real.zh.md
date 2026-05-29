# 如何为 Sim2Real 加固策略

部署配方：**训练时该启用哪些域随机化（DR）和噪声、导出时该 dump 什么、
部署端该如何对齐策略。** 假设已有可训练和回放的 task（见《运行 RL 实验》）。

目标是让策略在不接触真机的前提下，扛住现实差距——不准的惯量、摩擦、PD 增益、传感器
偏置和执行器缺陷。

## 1. 训练时启用域随机化

DR 事件挂在 `events_cfg` 上，遵循 [`EventTermCfg`](../reference/configuration.md) 的调用约定。
用 `mode="startup"` 每个 episode **采样一次**；用 `mode="interval"` 在 **episode 中途**
采样。

| DR 事件（`genelab.mdp.dr` / `mdp.events`） | 扰动对象 |
|---|---|
| `geom_friction` | 每个 link 的摩擦系数（地面 / 脚） |
| `body_com_offset` | 质心位置（惯量标定误差） |
| `body_mass_offset` | 每个 link 的质量（kg） |
| `randomize_joint_stiffness_damping` | 每-env PD 增益（kp / kv）——implicit-PD 走 sim 侧，force 通道走 Python 增益缩放 |
| `randomize_actuator_deadzone` | 每关节力矩死区（静摩擦 / 回差） |
| `encoder_bias` | 关节角恒定偏置（被悄悄移动的零点） |
| `push_by_setting_velocity`（`mdp.events`） | 脉冲式基座速度冲击——用 `mode="interval"` |

一段代表性的 startup 配置（对照 `examples/unitree/.../g1/env_cfg.py`）：

```python
from genelab import mdp
from genelab.managers import EventTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg

events_cfg = {
    "foot_friction": EventTermCfg(
        mode="startup",
        func=mdp.dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot")),
                "ranges": (0.3, 1.2), "shared_random": True},
    ),
    "base_com": EventTermCfg(
        mode="startup",
        func=mdp.dr.body_com_offset,
        params={"asset_cfg": SceneEntityCfg(name="robot", link_names=("torso_link",)),
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}},
    ),
    "pd_gains": EventTermCfg(
        mode="startup",
        func=mdp.dr.randomize_joint_stiffness_damping,
        params={"stiffness_range": (0.8, 1.2), "damping_range": (0.8, 1.2)},
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=mdp.dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "bias_range": (-0.015, 0.015)},
    ),
    # 每 5–10 s 仿真时间推一次。
    "push": EventTermCfg(
        mode="interval",
        interval_range_s=(5.0, 10.0),
        func=mdp.events.push_by_setting_velocity,
        params={"velocity_range": {0: (-0.5, 0.5), 1: (-0.5, 0.5)}},
    ),
}
```

从温和的范围起步，逐步加宽，直到同 seed 的 return 下降不超过约 10%——这正是《Reference Runs》
验收用的预算。

## 2. 像真实传感器那样污染观测

在 **policy** 观测组上设 `enable_corruption=True`，并给每个 term 挂一个 `NoiseCfg`。让
**critic** 组保持无污染——它从干净状态学习。

| `genelab.mdp.noise` 模型 | 用途 |
|---|---|
| `Unoise` / `Gnoise` | 基线加性传感器噪声 |
| `ScaledNoise` | 比例 / 增益误差（随信号增大） |
| `CorrelatedNoise` | 时间相关（AR(1)）噪声——缓慢的有色漂移 |
| `BiasDrift` | 缓慢漂移的偏置（随机游走，可选裁剪） |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import BiasDrift, CorrelatedNoise, Unoise

policy = ObservationGroupCfg(
    enable_corruption=True,
    terms={
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=CorrelatedNoise(std=0.5, alpha=0.8)),
        "imu_ang_vel": ObservationTermCfg(func=mdp.sensor_data,
                                          params={"sensor_name": "imu"}, noise=BiasDrift(drift_std=0.002)),
    },
)
critic = ObservationGroupCfg(enable_corruption=False, terms=policy.terms)
```

IMU 传感器自带每-env 偏置——在 `IMUSensorCfg` 上设 `bias_range_lin_acc` /
`bias_range_ang_acc`（每次 reset 重采样），而不是再叠一层 `BiasDrift`。

> `CorrelatedNoise` / `BiasDrift` 是**有状态**的，且故意**不**在 episode 边界重置
> （真实的漂移传感器并不知道 reset）。

## 3. 让策略尊重真实执行器极限

加上加固项，让策略不会学到硬件无法复现的行为。

* **Termination**（`genelab.mdp`）：`joint_pos_out_of_limit`、`joint_vel_out_of_limit`、
  `contact_force_limit(sensor_name, max_force)`。用 `ArticulationCfg.joint_vel_limit` 设一个软
  速度上限，`joint_vel_out_of_limit` 和 `joint_vel_limits` 奖励才会生效。
* **Reward**（`genelab.mdp`）：`applied_torque_l2`（罚力矩）、`joint_vel_limits`（罚超速）、
  `alive_bonus`（抵消每步惩罚）、`lin_vel_z_l2` / `base_height_l2`（抑制弹跳）。

## 4. 建模执行器差距（可选）

有真机力矩跟踪日志时，用 `MlpResidualActuator`——`DCMotorActuator` 基座加一个作用在
`[pos_error, joint_vel]` 上的 TorchScript 残差。残差网络在下游训练，把
`MlpResidualActuatorCfg.network_file` 指向保存的 `.pt`。不给文件时退化为普通
`DCMotorActuator`。

```python
from genelab.actuator import MlpResidualActuatorCfg

robot_cfg.actuators["drive"] = MlpResidualActuatorCfg(
    target_names_expr=(".*_joint",),
    stiffness=35.0,
    damping=0.8,
    effort_limit=80.0,
    velocity_limit=25.0,
    saturation_effort=80.0,
    network_file="assets/actuators/drive_residual.pt",
    residual_scale=0.25,
)
```

TorchScript 的输入/输出约定和字段说明见 [执行器](../concepts/actuators.md)。

## 5. 导出无依赖的策略

```bash
genelab export TASK_ID logs/.../model_best.pt --format onnx --out policy.onnx
```

导出会写出 `policy.onnx`（纯 `nn.Module`，无 rsl_rl/skrl/sb3）**以及**
`policy.onnx.metadata.json`。导出模型**内置了 obs 的 scale + clip**，因此它接收**原始**观测、
输出动作。`metadata.json` 按 obs 组记录：

* `dim` 和有序的 `terms`（每个 term 的 `name`、维度、`scale`、`clip`）；
* 动作 `dim` / 范围；
* 溯源信息（`task`、`checkpoint`）。

## 6. 对齐部署端

硬件控制器**该做**和**不该做**的：

* **按 `metadata.json` 的 term 顺序拼装 obs 向量**，拼接成**原始**值（不要预先 scale——
  导出模型内部会做 `scale`/`clip`）。
* **从动作重建关节目标**：`target = default_joint_pos + action_scale * action`，用与 env 执行器
  相同的 `action_scale` 和默认姿态。导出模型只输出原始动作。
* **不要在硬件上复现仅限仿真的污染**——DR（§1）、观测噪声（§2）和 `encoder_bias` 是*训练*
  扰动。真机本身已有自己的摩擦、偏置和传感器噪声；再叠加仿真版本会把差距翻倍。

## 7. 上线前验证

* `genelab eval TASK_ID model_best.pt --episodes 100` 拿确定性的 return / success-rate 数字，
  与《Reference Runs》对比。
* 把 `metadata.json` 的 obs term 顺序和硬件 obs 拼装做 diff——顺序或 scale 错位的 obs 向量
  是最常见的隐性部署故障。
* 在纯 `torch` 进程里抽查导出模型（加载、喂零 obs、确认动作形状与 `metadata.json` 一致）。
