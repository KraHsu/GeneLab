# 传感器

Genesis 不解析 MJCF 中的 `<sensor>` 段，因此 GeneLab 提供了一套 backend-agnostic 的传感器抽象。
接口与 mjlab 的 `SensorCfg` / `Sensor[T]` 对齐，每个具体传感器从 env 的 `RobotState` 而非
MuJoCo sensordata 取数 —— 观测与奖励 term 可在两套后端之间无缝迁移。

## 生命周期

`bind(env)` 在构造时调用一次。每个控制步 `update(dt)` 失效缓存；首次访问 `data` 触发
`_compute_data` 惰性求值；`reset(env_ids)` 同样失效缓存，并允许有状态传感器清空对应 env 的
缓冲区。

```python
class Sensor[T](ABC):
    def bind(self, env: "ManagerBasedRlEnv") -> None: ...
    @property
    def data(self) -> T: ...
    def update(self, dt: float) -> None: ...
    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...
    @abstractmethod
    def _compute_data(self) -> T: ...
```

env 自动接好生命周期：`SceneCfg.sensors` 里构造出的传感器，`update` 在 `_refresh_robot_state`
之后被调用，`reset` 在 `_reset_idx` 内部被调用 —— 奖励和观测 term 总能读到当前步的数据。

## 注册到 scene

`SceneCfg.sensors` 是一个 `SensorCfg` 元组。`ManagerBasedRlEnv.__init__` 对每个 cfg 调用
`build()` 并把生成的传感器绑定到 env。运行时通过 `env.sensors[name].data` 访问。

```python
from genelab.configs import SceneCfg
from genelab.sensor import BodyVelocitySensorCfg, ContactSensorCfg

scene = SceneCfg(
    num_envs=4096,
    sensors=(
        BodyVelocitySensorCfg(
            name="imu_lin_vel",
            link_name="pelvis",
            offset=(0.04525, 0.0, -0.08339),
            measure="lin_vel",
        ),
        ContactSensorCfg(
            name="feet_ground_contact",
            link_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            track_air_time=True,
        ),
    ),
)
```

## 内置传感器

### BodyVelocitySensor

对应 MuJoCo 的 `<velocimeter>` / `<gyro>`：在与 link 刚性连接的 site 上读取线速度或角速度，
结果旋转到 link body frame。支持杠杆臂偏移与可选的每 env 均匀分布偏置。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_name` | `str` | site 刚性连接到的 link。 |
| `offset` | `tuple[float, float, float]` | site 在 link 局部系下的位置，`ang_vel` 模式忽略。 |
| `measure` | `Literal["lin_vel", "ang_vel"]` | 选择 velocimeter（线速度）或 gyro（角速度）。 |
| `bias_range` | `tuple[float, float] \| None` | 每 env 均匀偏置，`reset` 时重采样。 |

velocimeter 计算公式为 `v_site = v_link + ω × (R_link · offset)`，再旋转到 body frame ——
与 MuJoCo 的杠杆臂约定一致，使得 GeneLab 训出的策略读到的信号与 mjlab 参考实现完全相同。

### ContactSensor

按 link 名汇总 `robot.get_links_net_contact_force()` 输出。开启 `track_air_time=True` 时，
一个内部状态机会推进每 env 的 `current_air_time` / `current_contact_time`，并在接触状态翻转
的瞬间把已完成的时长快照到 `last_air_time` / `last_contact_time`。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_names` | `tuple[str, ...]` | 显式列出需要监控的 link。 |
| `link_names_expr` | `str \| None` | 对 `env.link_names` 做正则匹配。 |
| `force_threshold` | `float` | 力幅值（N）大于此值时 `found` 为真。 |
| `track_air_time` | `bool` | 是否分配 air-time / contact-time 状态机。 |

`data` 是 `ContactData` 数据类，包含 `force`、`force_norm`、`found` 以及四个 air-time 缓冲。
配套的观测 term —— `mdp.foot_air_time`、`mdp.foot_contact`、`mdp.foot_contact_forces` ——
直接读取这个数据类。

### TerrainHeightSensor

锚定在某个 link 上的 2D 下射光线网格，输出每条光线相对地形的高度（正值表示在地形上方），
适合作为 critic 的 privileged `height_scan` 观测。默认 backend 把每条光线打到位于
`ground_height` 的水平面上；非平地场景的扩展点是继承 `RayCastSensor` 并重载
`_intersect_world_rays`。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_name` | `str` | 网格原点锚定的 link。 |
| `pattern` | `GridPattern` | 网格分辨率 / 尺寸 / 方向。 |
| `attach_yaw_only` | `bool` | 仅按 yaw 旋转网格，使其保持水平。 |
| `max_distance` | `float` | 光线距离上限。 |
| `ground_height` | `float` | 默认平面 backend 使用的地面高度。 |

## 给观测加噪

`ObservationTermCfg.noise` 接收加性噪声模型（`Unoise(n_min, n_max)` 或
`Gnoise(mean, std)`）。`ObservationGroupCfg.enable_corruption` 控制是否真正加噪 —— 默认关闭，
让 critic 看到 ground truth。逐 term 的管线顺序是 **noise → scale → clip**，因此噪声幅值
定义在原始信号空间：在带 `scale=0.05` 的原始 `joint_vel` 上加 `Unoise(-1.5, 1.5)`，最终的
抖动量为 ±0.075。

惯例做法是 policy / critic 共享 term，只在 `enable_corruption` 上分叉：

```python
from genelab import mdp
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import Unoise


def _obs_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_lin_vel"},
            noise=Unoise(-0.5, 0.5),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(-1.5, 1.5),
        ),
    }


policy = ObservationGroupCfg(enable_corruption=True, terms=_obs_terms())
critic = ObservationGroupCfg(enable_corruption=False, terms=_obs_terms())
```

## 自定义传感器

继承 `Sensor[T]` 并指定返回类型，实现 `_compute_data`。如需缓存 link 索引、推进积分器或
清空 env 状态，重载 `bind` / `update` / `reset` 并先调用 `super()`，让缓存失效链路保持完整。

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class JointTorqueSensorCfg(SensorCfg):
    def build(self) -> "JointTorqueSensor":
        return JointTorqueSensor(self)


class JointTorqueSensor(Sensor[torch.Tensor]):
    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        # 在这里缓存依赖 env.link_names / env.joint_names 的索引。

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None
        rs = self._env.robot_state
        return self._env.joint_kp * (rs.joint_pos - self._env.default_joint_pos)
```

把 cfg 加入 `SceneCfg.sensors` 后，传感器即可通过 `env.sensors[name].data` 访问。

## See also

- [配置系统](configs.md)
- [API 参考](../api/reference.md)
