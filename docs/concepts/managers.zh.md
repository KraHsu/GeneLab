# Manager 与 MDP term

`genelab.managers` 暴露了 `ManagerBasedRlEnv` 编排的七个 manager 风格 MDP 钩子：
actions、commands、observations、rewards、terminations、events、curriculums。
每个 manager 持有一个按名键入的 term cfg 字典；env 按固定顺序构造它们，再按每控制
步或每次 reset 的节奏驱动。

## term 与 manager

*term* 是单条 MDP 侧的计算 —— 一个 reward 分量、一组 observation 列、一个终止判定。
*manager* 是容器，负责实例化某一类 term、暴露聚合输出、并按每步 / 每次 reset 调度。

```
ManagerBasedRlEnvCfg
├── actions_cfg:       dict[str, ActionTermCfg]
├── commands_cfg:      dict[str, CommandTermCfg]
├── observations_cfg:  dict[str, ObservationGroupCfg]   # group → terms
├── rewards_cfg:       dict[str, RewardTermCfg]
├── terminations_cfg:  dict[str, TerminationTermCfg]
├── events_cfg:        dict[str, EventTermCfg]
└── curriculum_cfg:    dict[str, CurriculumTermCfg]
```

所有 term cfg 都继承 `ManagerTermBaseCfg`，含两个字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `func` | `Callable[..., Any]` | 每步执行的计算。observation / reward / termination / event / curriculum 类 term 的 `func` 签名为 `(env, **params)` 并返回 tensor（event 可返回 `None`）。action 与 command term 不使用 `func`，改用 `class_type`。 |
| `params` | `dict[str, Any]` | 每次调用注入的关键字参数。`apply_overrides` 可按 dotted path 改写其中任一条目。 |

## 七个 manager

| Manager | Term cfg | term 识别字段 | 调用节奏 |
|---|---|---|---|
| `ActionManager` | `ActionTermCfg(class_type=…)` | `class_type` | 每 env step 一次 `process_action`；每 env step 内 `apply_action` 触发 `decimation` 次 |
| `CommandManager` | `CommandTermCfg(class_type=…)` | `class_type` | 每 env step 一次 `compute(dt)`（按倒计时重采样） |
| `ObservationManager` | `ObservationGroupCfg.terms[name] = ObservationTermCfg(func=…)` | `func` | 每 env step 一次 `compute()`（reset 后与 step 后各一次） |
| `RewardManager` | `RewardTermCfg(func=…, weight=…)` | `func` | 每 env step 一次 `compute(dt)` |
| `TerminationManager` | `TerminationTermCfg(func=…, time_out=…)` | `func` | 每 env step 一次 `compute()` |
| `EventManager` | `EventTermCfg(func=…, mode=…)` | `func` | 构造时一次（`startup`）、每次 reset 一次（`reset`）、按每 env 倒计时触发（`interval`） |
| `CurriculumManager` | `CurriculumTermCfg(func=…)` | `func` | 每次 reset 一次 `compute(env_ids)`，在其他 manager reset 之后 |

## term 生命周期

`ManagerBasedRlEnv.__init__` 按上表顺序构造七个 manager。每个 term 的处理：

1. Manager 深拷贝 cfg 字典，避免逐实例 mutation 反向污染源 cfg。
2. 当 `func` 是类时，`_base.instantiate_class_term` 把它替换为
   `func(cfg=term_cfg, env=env)`，让 term 在构造期缓存引用。普通可调用对象保持原样。
3. action / command manager 改用 `class_type(term_cfg, env)`，把得到的实例按 term 名存好。

七个 manager 全部构造完毕后，env 调用一次 `event_manager.apply("startup")`，
再 `articulation.refresh()`，最后对所有 env 跑一次 `reset()`。

## ActionManager

每个 `ActionTermCfg` 声明一个 `class_type`（`ActionTerm` 子类）与可选的 `asset_name`。
Manager 把每个 term 的 `action_dim` 拼成单个扁平 action 向量。多个 action term
可以并存 —— 例如手臂一个、夹爪一个：

```python
from genelab.mdp.actions.joint_position import JointPositionActionCfg

actions_cfg = {
    "panda_arm": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"^joint[1-7]$",),
        use_default_offset=True,
    ),
    "panda_hand": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"finger_joint.*",),
        use_default_offset=True,
    ),
}
```

`process_action(action)` 在每 env step 调用一次；`apply_action()` 在每 env step 内
触发 `decimation` 次（每个 sim 子步前一次），因此控制器在整个 decimation 窗口内
看到的是同一个目标。

## ObservationManager

Observation 按命名 *group*（`policy` / `critic` / …）组织；每个 group 是命名 term 字典。

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab import mdp

observations_cfg = {
    "policy": ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
            "last_action": ObservationTermCfg(func=mdp.last_action),
        },
        concatenate_terms=True,
        enable_corruption=False,
    ),
}
```

`compute()` 对每个 term 调用 `term_cfg.func(env, **term_cfg.params)`，1-D 返回会被
自动 unsqueeze 成 `(num_envs, 1)`，随后按需 apply `noise`（仅当
`enable_corruption=True`）、`scale`、`clip`。`concatenate_terms=True`（默认）时整组
输出是单张 `(num_envs, total_dim)` tensor；否则 manager 沿新末轴 stack。

## RewardManager

`RewardTermCfg.weight` 是带符号标量乘子。Manager 跳过 weight=0 的 term。当
`ManagerBasedRlEnvCfg.scale_rewards_by_dt=True`（默认）时，逐 term tensor 还会再乘
env step `dt`，使得 episode 总回报在不同仿真频率下可比。

```python
from genelab.managers import RewardTermCfg
from genelab import mdp

rewards_cfg = {
    "track_lin_vel": RewardTermCfg(
        func=mdp.track_linear_velocity_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "joint_acc":   RewardTermCfg(func=mdp.joint_acc_l2,   weight=-2.5e-7),
}
```

逐 term 的 `weight * func(...)` 结果会 NaN coerce、求和，并作为 `(num_envs,)` 的
reward tensor 输出。Manager 同时累计每 term 的 episode 求和；`reset()` 返回每
term 的 episode 平均奖励（按 `Episode_Reward/<name>` 写入 `extras["log"]`）。

## TerminationManager

每个 `TerminationTermCfg.func` 必须返回 `(num_envs,)` 的 bool tensor。`time_out=True`
的 term 写入 truncation buffer（`info["time_out"]`），其余写入 terminated buffer。
两者 OR 后得到最终 `dones`。RSL-RL 区分这两者以在 PPO 更新中正确处理截断。

```python
from genelab.managers import TerminationTermCfg
from genelab import mdp

terminations_cfg = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    ),
}
```

## EventManager

`EventTermCfg.mode` 是 `Literal["startup", "reset", "interval"]`：

| Mode | 触发时机 |
|---|---|
| `startup` | 全部 manager 构造完成后一次。适合一次性随机化物理参数（质量、摩擦）。 |
| `reset` | 每次 env reset，在 `command_manager.reset` 与 `action_manager.reset` 之前。适合随机化初始 joint state、root pose、env 缓存。 |
| `interval` | 按每 env 倒计时（从 `interval_range_s` 均匀采样）。在 env step 内倒计时归零时触发，触发后重新采样。适合周期性扰动（随机推力）。 |

```python
from genelab.managers import EventTermCfg
from genelab import mdp

events_cfg = {
    "reset_joints": EventTermCfg(
        mode="reset",
        func=mdp.reset_joints_to_default,
        params={"pos_jitter": 0.05, "vel_jitter": 0.0},
    ),
    "push_robot": EventTermCfg(
        mode="interval",
        func=mdp.push_by_setting_velocity,
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
}
```

## CommandManager

*command* 是按 env 持久化的目标（速度命令、动作参考、目标位姿），跨 env step 保留，
直到被重采样。每个 `CommandTermCfg.class_type` 是 `CommandTerm` 子类；term 持有
当前 tensor 并每 `resampling_time_range` 秒重采样一次。observation 与 reward term
通过 `env.command_manager.get_command(name)` 读回。

```python
from genelab.mdp.commands.uniform_velocity import UniformVelocityCommandCfg

commands_cfg = {
    "base_velocity": UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    ),
}
```

## CurriculumManager

Curriculum term 在每次 reset 触发（在其他 manager reset 之后），可改写 scene 状态 ——
`TerrainImporter.terrain_levels`、env 专属 spawn pose、reward weight。返回值按
`Curriculum/<name>` 写入 `extras["log"]`；tensor 返回会被 reduce 成在 `env_ids` 上的均值。

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp.curriculums import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={
            "command_name": "base_velocity",
            "distance_threshold": 5.0,
            "demote_ratio": 0.5,
        },
    ),
}
```

## 已知失效模式

!!! warning "observation tensor 形状错误"
    `ObservationManager.compute` 只会把 1-D 返回 auto-unsqueeze 成 `(num_envs, 1)`。
    意外返回 `()` 标量的 reward term 会朝错误方向广播。term 必须返回 `(num_envs,)`
    或 `(num_envs, d)`。

!!! warning "缺失 class_type / func"
    `ActionManager` 与 `CommandManager` 会静默跳过 `class_type=None` 的 term。
    `RewardManager` / `TerminationManager` / `EventManager` / `CurriculumManager`
    保留 `ManagerTermBaseCfg` 的 no-op 默认 `func`，缺失 `func=` 会构造出返回 `None`
    的 manager，并在下游把结果当 tensor 用时崩溃。term factory 必须显式设置。

!!! warning "interval event 缺 `interval_range_s`"
    `EventManager.__init__` 断言每个 `mode="interval"` term 必须给 `interval_range_s=(low, high)`。
    断言在 manager 构造时触发，错配的 task 永远跑不到第一次 reset。

!!! tip "在 CLI 覆盖 term 参数"
    每个 term 的 `params` 条目都可通过 `apply_overrides` 访问。dotted path 与
    `genelab info <task-id>` 打印的 cfg 树一一对应：
    `--env.rewards_cfg.track_lin_vel.weight 2.0` 重新缩放一个 reward term；
    `--env.rewards_cfg.track_lin_vel.params.std 0.5` 改写一个 kwarg。

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
- [Discovery: list and info](../cli/list-info.md)
