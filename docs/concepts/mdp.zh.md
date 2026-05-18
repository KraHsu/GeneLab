# MDP term 参考

`genelab.mdp` 是一组可直接插入七个 manager 的 Python 可调用对象。所有函数遵循相同
签名形状 —— observation / reward / termination 是 `(env, **params) -> torch.Tensor`，
event 是 `(env, env_ids, **params) -> None`，curriculum 是
`(env, env_ids, **params) -> Tensor` —— 并由相应 `*TermCfg.func`（或 `class_type`）
字段引用。

## 可复用 building block

下表列出的所有名字都已在 `genelab.mdp.__init__` 中 re-export，因此
`from genelab import mdp` 就能拿到。完整模块结构：

```
genelab.mdp.actions       → JointPositionAction(Cfg)
genelab.mdp.commands      → UniformVelocityCommand(Cfg)、MotionCommand(Cfg)、MotionLoader
genelab.mdp.observations  → 16 个 observation 函数
genelab.mdp.rewards       → 15 个 reward 函数 / 类
genelab.mdp.terminations  → 6 个 termination 函数
genelab.mdp.events        → 3 个 event 函数
genelab.mdp.curriculums   → 1 个 curriculum 函数
genelab.mdp.noise         → NoiseCfg 基类 + Unoise + Gnoise
```

下面 shape 列里 `B = num_envs`、`D = action / joint / sensor 维度`、
`N = 身体段或脚的个数`。reward 与 termination term 总是返回 `(B,)`；observation
term 返回 `(B, D)`（或 `(B,)`，会被 `ObservationManager` 自动 unsqueeze 成 `(B, 1)`）。

## Actions

| 名字 | 签名（用到的 Cfg 字段） | 行为 |
|---|---|---|
| `JointPositionAction` | `JointPositionActionCfg(joint_names, scale, use_default_offset, asset_name)` | `target = default + scale * raw_action`。目标通过 `Articulation.write_joint_targets_partial` 派发，按每个 joint 所在 actuator group 声明的通道执行（implicit PD 走 position，`IdealPDActuator` / `DCMotorActuator` 走 force）。`scale=None`（默认）从 `Articulation.action_scale_tensor` 继承逐 joint 比例；`float` 或 `dict[str, float]` 可覆盖。`joint_names` 为 regex，匹配不到任何 joint 会抛错。 |

```python
from genelab.mdp.actions.joint_position import JointPositionActionCfg

actions_cfg = {
    "panda_arm": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"^joint[1-7]$",),
        use_default_offset=True,
    ),
}
```

## Commands

| 名字 | 签名（用到的 Cfg 字段） | 行为 |
|---|---|---|
| `UniformVelocityCommand` | `UniformVelocityCommandCfg(ranges, rel_standing_envs, heading_command, heading_control_stiffness, resampling_time_range, asset_name)` | 每 env 一组 body-frame `[lin_vel_x, lin_vel_y, ang_vel_z]`，每 `resampling_time_range` 秒从 `ranges` 均匀重采样。`heading_command=True`（默认）时，`ang_vel_z` 每步被重写以把 body 朝向目标 heading 拉。`rel_standing_envs` 把一定比例的重采样命令清零，让 policy 也见到 "stand still" 目标。`command` shape `(B, 3)`。 |
| `MotionCommand` | `MotionCommandCfg(motion_file, anchor_body_name, body_names, motion_body_order, motion_joint_order, pose_range, velocity_range, sampling_mode, ...)` | 按帧驱动 env 跟随预录动作 clip。读 mjlab `csv_to_npz` 产生的 NPZ schema。`motion_body_order` / `motion_joint_order` 描述 NPZ 内部 body / joint 轴的原生顺序（通常是 mjlab MJCF 的 DFS），`MotionLoader` 据此把数据重排到运行时机器人的顺序 —— 上游遍历顺序与 Genesis 不一致时必须传入。暴露 anchor 与多 body 的 reference / current 位姿 / 速度，供下节相对位姿 reward 函数使用。`sampling_mode="start"` 始终从第 0 帧开始；`"uniform"`（默认）随机抽帧。 |
| `MotionLoader` | `MotionLoader(motion_file, body_indexes, device, joint_perm=None)` | `MotionCommand` 内部用的辅助类。把 NPZ 读入设备 tensor，按可选的 `joint_perm` 重排 joint 轴，并把 `body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w` 切到要跟踪的几个 body。直接使用极少 —— 通常实例化 `MotionCommandCfg` 即可。 |

```python
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg

commands_cfg = {
    "base_velocity": UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
        heading_command=True,
    ),
}
```

## Observations

下列所有函数都接 `env` 加注明的 kwargs，并返回 shape `(B, D)` 或 `(B,)` 的 tensor
（后者会被 manager auto-unsqueeze 成 `(B, 1)`）。

| 名字 | Params | Shape | 来源 |
|---|---|---|---|
| `base_lin_vel` | — | `(B, 3)` | body-frame 浮动基线速度。 |
| `base_ang_vel` | — | `(B, 3)` | body-frame 浮动基角速度。 |
| `projected_gravity` | — | `(B, 3)` | 世界重力投影到 body frame（IMU 朝向 proxy）。 |
| `joint_pos_rel` | — | `(B, num_dofs)` | 关节位置减默认 pose。 |
| `joint_vel_rel` | — | `(B, num_dofs)` | 关节速度（默认为零，等同 raw `joint_vel`）。 |
| `last_action` | — | `(B, total_action_dim)` | 上一步 manager 处理的 action tensor。 |
| `generated_commands` | `command_name` | `(B, command_dim)` | `env.command_manager.get_command(command_name)`。 |
| `sensor_data` | `sensor_name` | 视 sensor 而定 | `env.sensors[sensor_name]` 上每步缓存的 tensor。适用于 IMU、FrameTransformer、ray-cast 等输出已是 tensor 的 sensor。 |
| `foot_air_time` | `sensor_name`（必须是 `ContactSensor`） | `(B, N_feet)` | 每脚的当前 air time（接触时为零）。 |
| `foot_contact` | `sensor_name`（必须是 `ContactSensor`） | `(B, N_feet)` | 接触二值 mask（float）。 |
| `foot_contact_forces` | `sensor_name`（必须是 `ContactSensor`） | `(B, N_feet * 3)` | 每脚接触力，按 `sign(f) * log1p(|f|)` 压缩并铺平。 |
| `height_scan` | `sensor_name`（必须是 `TerrainHeightSensor`） | `(B, num_rays)` | 每条射线相对地形的高度。 |
| `motion_anchor_pos_b` | `command_name`（必须是 `MotionCommand`） | `(B, 3)` | 参考 anchor 位置在 robot anchor frame 下的表示。 |
| `motion_anchor_ori_b` | `command_name` | `(B, 6)` | 参考 6D 朝向（rotation matrix 前两列）。 |
| `robot_body_pos_b` | `command_name` | `(B, N_bodies * 3)` | 每身体段在 robot anchor frame 下的位置（critic 用的特权观测）。 |
| `robot_body_ori_b` | `command_name` | `(B, N_bodies * 6)` | 每身体段在 robot anchor frame 下的 6D 朝向。 |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab import mdp
from genelab.mdp.noise import Unoise

observations_cfg = {
    "policy": ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel,
                                                noise=Unoise(n_min=-0.1, n_max=0.1)),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
            "last_action": ObservationTermCfg(func=mdp.last_action),
            "commands": ObservationTermCfg(func=mdp.generated_commands,
                                            params={"command_name": "base_velocity"}),
        },
        enable_corruption=True,
    ),
}
```

## Rewards

所有 reward 函数返回 shape `(B,)`。`RewardManager` 会乘 `RewardTermCfg.weight`，并在
`scale_rewards_by_dt=True` 时再乘 `dt`。负 weight 把"偏差"型 reward 变成惩罚。

| 名字 | Params | 行为 |
|---|---|---|
| `track_linear_velocity_xy_exp` | `command_name`、`std=0.5` | `exp(-||cmd_xy − vel_xy||² / std²)`。 |
| `track_angular_velocity_z_exp` | `command_name`、`std=0.5` | `exp(-(cmd_z − vel_z)² / std²)`。 |
| `action_rate_l2` | — | `Σ_d (action_d − prev_action_d)²` —— 惩罚 action 抖动。 |
| `joint_acc_l2` | — | **占位**（当前返回零；待加 accel buffer）。 |
| `flat_orientation_l2` | — | `Σ (projected_gravity_xy)²` —— 惩罚倾斜。 |
| `upright_exp` | `std=0.45` | `exp(-||projected_gravity_xy||² / std²)` —— 站直时正 reward。 |
| `variable_posture` | `command_name`、`std_standing` / `std_walking` / `std_running`（regex → float 字典）、`default_std`、`walking_threshold`、`running_threshold` | 速度自适应姿态 reward：`exp(-mean((joint_pos − default)² / std²))`，每 env 按命令模长挑 std。class 风格 term —— 构造时一次性 `__init__`，每步走 `__call__`。 |
| `joint_pos_limits` | — | 关节位置超过 ±π 部分的 L2。 |
| `feet_air_time` | `threshold=0.4` | **stub** —— 用 foot link 平均高度近似 air-time，超过 `threshold` 饱和。 |
| `motion_global_anchor_position_error_exp` | `command_name`、`std` | 世界系 anchor 位置 `exp(-||p_ref − p_robot||² / std²)`。 |
| `motion_global_anchor_orientation_error_exp` | `command_name`、`std` | 世界系 anchor 朝向几何 rotation error 套 Gaussian 核。 |
| `motion_relative_body_position_error_exp` | `command_name`、`std`、`body_names=None` | anchor-aligned 多 body 位置 L2 误差。`body_names=None` 表示所有被跟踪 body。 |
| `motion_relative_body_orientation_error_exp` | `command_name`、`std`、`body_names=None` | anchor-aligned 多 body 几何 rotation 误差。 |
| `motion_global_body_linear_velocity_error_exp` | `command_name`、`std`、`body_names=None` | 世界系多 body 线速度 L2 误差。 |
| `motion_global_body_angular_velocity_error_exp` | `command_name`、`std`、`body_names=None` | 世界系多 body 角速度 L2 误差。 |

!!! warning "占位 reward"
    `joint_acc_l2` 与 `feet_air_time` 是 stub，等正经 accel / 接触缓冲落地。它们能正常
    返回 shape 正确的 tensor，但**值本身不是**通常意义下的 locomotion 信号。除非已经
    把它们接到逐步速度历史（`joint_acc_l2`）与 `ContactSensor`（`feet_air_time` ——
    `mdp.foot_air_time` 作为 observation term 已经能给真实信号），否则按 `weight=0.0`
    占位处理。

```python
from genelab.managers import RewardTermCfg
from genelab import mdp

rewards_cfg = {
    "track_lin_vel": RewardTermCfg(
        func=mdp.track_linear_velocity_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "track_ang_vel": RewardTermCfg(
        func=mdp.track_angular_velocity_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "upright":       RewardTermCfg(func=mdp.upright_exp,         weight=0.2),
    "action_rate":   RewardTermCfg(func=mdp.action_rate_l2,      weight=-0.005),
    "flat_ori":      RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5),
}
```

## Terminations

所有 termination 函数返回 shape `(B,)` bool。`TerminationTermCfg.time_out` 把返回值
送进 truncation buffer（RSL-RL 的 `info["time_out"]`），而不是 terminated buffer。

| 名字 | Params | 行为 |
|---|---|---|
| `time_out` | — | `episode_length_buf >= max_episode_length`。永远配 `time_out=True`。 |
| `bad_orientation` | `limit_angle=math.radians(70.0)` | body z 轴相对世界向上方向倾斜超过 `limit_angle` 时为 True。 |
| `root_height_below` | `min_height` | `root_pos.z < min_height` 时为 True。 |
| `bad_anchor_pos_z_only` | `command_name`、`threshold` | robot anchor z 相对 reference clip 漂移超过 `threshold` 时为 True。 |
| `bad_anchor_ori` | `command_name`、`threshold` | robot anchor 与 reference 倾斜误差（重力 z 投影 proxy）超过 `threshold` 时为 True。 |
| `bad_motion_body_pos_z_only` | `command_name`、`threshold`、`body_names=None` | 任一选中 body 的垂直位置偏离参考超过 `threshold` 时为 True。 |

```python
from genelab.managers import TerminationTermCfg
from genelab import mdp

terminations_cfg = {
    "time_out":  TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation,
                                     params={"limit_angle": 1.0}),
}
```

## Events

Event 函数多接一个 `env_ids` 参数，返回 `None`。`EventTermCfg.mode` 决定触发时机 ——
`startup` 构造期一次、`reset` 每次 env reset、`interval` 每 env 倒计时（需要
`interval_range_s`）。

| 名字 | Params | 行为 |
|---|---|---|
| `reset_root_state_uniform` | `pose_range`（`x` / `y` / `z` / `roll` / `pitch` / `yaw` → `(low, high)` 字典）、`velocity_range`（同样的轴） | 在给定范围内随机化浮动基位姿与速度。位姿偏移叠加在 `cfg.robot.init_pos` 上；朝向偏移叠加在 `cfg.robot.init_quat` 上。 |
| `reset_joints_to_default` | `pos_jitter=0.0`、`vel_jitter=0.0` | 把默认关节 pose 写入选中 env，可选叠加位置与速度的 uniform ±jitter。 |
| `push_by_setting_velocity` | `velocity_range`（`x` / `y` / `z` / `roll` / `pitch` / `yaw` → `(low, high)`） | 覆盖 base 线速度与角速度。配合 `mode="interval"` 就是标准的 "随机推" 扰动。 |

```python
from genelab.managers import EventTermCfg
from genelab import mdp

events_cfg = {
    "reset_root": EventTermCfg(
        mode="reset",
        func=mdp.reset_root_state_uniform,
        params={"pose_range": {"yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (-0.1, 0.1)}},
    ),
    "push_robot": EventTermCfg(
        mode="interval",
        func=mdp.push_by_setting_velocity,
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
}
```

## Curriculums

| 名字 | Params | 行为 |
|---|---|---|
| `terrain_levels_vel` | `distance_threshold`、`demote_ratio=0.5` | 按每 env 离 spawn 走了多远，调整 `TerrainImporter.terrain_levels` 索引。走超过 `distance_threshold` 上一档；走少于 `distance_threshold * demote_ratio` 下一档。新 level 触发 spawn origin 重查，curriculum 把新 root pose 写回 sim。`env.scene.terrain is None` 时整体 no-op。返回每 env 平均 level，manager 自动按 `Curriculum/<term-name>` 写日志。 |

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp.curriculums import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"distance_threshold": 5.0, "demote_ratio": 0.5},
    ),
}
```

## Noise

`NoiseCfg` 是抽象基类；具体子类挂在 `ObservationTermCfg.noise` 上，仅当所在
`ObservationGroupCfg.enable_corruption=True` 时生效。

| 名字 | 字段 | 行为 |
|---|---|---|
| `NoiseCfg` | — | 抽象基类；继承并实现 `apply(data) -> Tensor`。 |
| `Unoise` | `n_min=-1.0`、`n_max=1.0` | uniform 噪声 `[n_min, n_max]`，加性叠加。 |
| `Gnoise` | `mean=0.0`、`std=1.0` | Gaussian 噪声 `N(mean, std²)`，加性叠加。 |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import Gnoise, Unoise
from genelab import mdp

observations_cfg = {
    "policy": ObservationGroupCfg(
        enable_corruption=True,
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel,
                                                noise=Unoise(n_min=-0.1, n_max=0.1)),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel,
                                                  noise=Gnoise(std=0.01)),
        },
    ),
}
```

## See also

- [Managers and MDP terms](managers.md)
- [Sensors](sensors.md)
- [API Reference](../api/reference.md)
