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

env 自动接好生命周期：`InteractiveSceneCfg.sensors` 里构造出的传感器，`update` 在每次
articulation 刷新之后被调用，`reset` 在 `_reset_idx` 内部被调用 —— 奖励和观测 term 总能读
到当前步的数据。

## 注册到 scene

`InteractiveSceneCfg.sensors` 是一个 `SensorCfg` 元组。`ManagerBasedRlEnv.__init__` 对每
个 cfg 调用 `build()` 并把生成的传感器绑定到 env。运行时通过 `env.sensors[name].data` 访
问。

```python
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.sensor import BodyVelocitySensorCfg, ContactSensorCfg

simulation = SimulationCfg(num_envs=4096)
scene = InteractiveSceneCfg(
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

### IMUSensor

刚性附着在某个 link 上的惯性测量单元，输出姿态（`link_quat_w`）、body frame 投影单位
重力、以及 body frame 下的线 / 角加速度。加速度通过对世界系速度做有限差分得到；每次
`reset` 之后的第一个控制步加速度恒为零，避免使用上一阶段残留的 `prev` 缓冲产生
伪冲击 —— 接近 reset 边界的 reward / observation term 在消费 `lin_acc_b` 或
`ang_acc_b` 时需要意识到这点。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_name` | `str` | IMU 刚性附着的 link。 |
| `offset` | `tuple[float, float, float]` | site 在 link 局部系下的位置；线加速度计算中携带杠杆臂项。 |
| `gravity_bias` | `bool` | 为 `True` 时输出比力 `R^T (a_w - g_w)`，与真实加速度计一致 —— 静止时读到 `+g` 沿向上轴。 |
| `bias_range_lin_acc` | `tuple[float, float] \| None` | 加速度计每 env 常数偏置，`reset` 时重采样。 |
| `bias_range_ang_acc` | `tuple[float, float] \| None` | 陀螺导出量每 env 常数偏置，`reset` 时重采样。 |

`IMUSensor` 不输出线速度或角速度 —— 这是 `BodyVelocitySensor` 的职责，二者在同一 link
上可以并存。控制步频率下的有限差分内部不做滤波；如需平滑，可在 `ObservationTermCfg`
上叠加 `noise` + `scale`，或者在观测 term 里自行做 EMA。

### CameraSensor

刚性挂载的 RGB-D 相机，通过 4×4 变换螺接到指定 link 上。`bind()` 解析 link，从
`scene.gs_scene.add_camera` 拿到一个 Genesis camera 句柄，根据 `offset_pos` /
`offset_quat` 构造偏置矩阵后调用 `cam.attach`。每次访问 `data` 都会先 `cam.move_to_attach()`
再 `cam.render(...)`，缓存的 :class:`CameraData` 携带 `rgb`（uint8）与/或 `depth`（米，float），
形状 `(num_envs, H, W, 3)` / `(num_envs, H, W)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_name` | `str` | 相机挂载到的 link（通过 `env.link_names` 解析）。 |
| `offset_pos` | `tuple[float, float, float]` | 相机原点在 link 局部系下的位置。 |
| `offset_quat` | `tuple[float, float, float, float]` | 相机相对 link 的姿态（wxyz）；Genesis 约定 `+x` 为前向。 |
| `width`、`height` | `int` | 像素分辨率；同一 scene 内所有 BatchRender 相机分辨率必须一致。 |
| `fov` | `float` | 垂直视场角（度）。 |
| `near`、`far` | `float` | 深度裁剪面（米）。 |
| `render_rgb`、`render_depth` | `bool` | 两条通道独立开关；关闭的那条在输出数据类里为 `None`。 |

!!! warning "BatchRenderer 是唯一的多 env 后端"
    多 env RGB-D 必须 `gs.init(backend=gs.cuda)` 配
    `gs.Scene(renderer=gs.renderers.BatchRenderer(use_rasterizer=False), ...)` ——
    仅 Linux x86-64 + CUDA 可用。macOS / 纯 CPU 环境下模块本身可正常 import，
    但 `bind()` 在 Genesis 分配相机句柄时就会抛错。

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

### FrameTransformerSensor

无状态的正向运动学探针：输出一个或多个目标系相对于单一源系的位姿，源 / 目标在各自
link 局部系下都支持刚性偏移。数据类同时暴露世界系位姿与源系位姿，下游 reward /
observation term 可按需取用而不必自行复合运算。典型场景：在 base link 系下读取末端执行器
位姿、对比两个脚尖系做步态约束、把负载位姿表达在抓取点局部系下。

| 字段 | 类型 | 含义 |
|------|------|------|
| `source_link_name` | `str` | 表达 target 所用的参考系。 |
| `source_offset_pos` | `tuple[float, float, float]` | 源端在 link 局部系下的位移偏移。 |
| `source_offset_quat` | `tuple[float, float, float, float]` | 源端旋转偏移（wxyz）。 |
| `target_frames` | `tuple[TargetFrameCfg, ...]` | 保序的 target 列表 —— 输出 `N` 轴顺序即 cfg 顺序。 |

每个 `TargetFrameCfg` 含 `link_name`、可选的 `name`（默认为 `link_name`）、`offset_pos`
与 `offset_quat`。传感器属性 `target_names: tuple[str, ...]` 暴露给下游按名字定位列。
`bind` 会拒绝未知 link 名以及空 `target_frames`。

### Ray-cast patterns

`RayCastSensorCfg.pattern` 接受三种内置 pattern 数据类。自定义 pattern 满足同一非正式协议
即可接入 —— `num_rays() -> int` 和 `generate(device) -> (starts, dirs)`，两个张量都为
`(M, 3)`，定义在传感器局部系下 —— 无须改动 `RayCastSensor` 本身。

`GridPattern` 是默认：所有光线平行，原点排在 2D 矩形上。`RingPattern` 从原点发出
`num_horizontal × num_vertical` 条光线，方位角和俯仰角各自均匀分布 —— 典型的多线 LIDAR
布局。`HemispherePattern` 用 Fibonacci 格点把 `num_rays_target` 条光线均匀分布到以
`pole_axis` 为极轴、半角为 `polar_fov_deg` 的球冠上；90° 覆盖整个半球，180° 即整个球面。

| Pattern | 关键字段 | 典型用途 |
|---------|----------|----------|
| `GridPattern` | `resolution`、`size`、`direction` | 高度场网格、面扫 |
| `RingPattern` | `num_horizontal`、`num_vertical`、`horizontal_fov_deg`、`vertical_fov_deg` | 平面 / 多线 LIDAR |
| `HemispherePattern` | `num_rays_target`、`pole_axis`、`polar_fov_deg` | 接近告警球罩、向下覆盖 |

`RingPattern` 将水平角恰好 ±360° 视为环绕扫描，自动丢弃首末重复方位；任意其它跨度（例如
`(-30, 30)` 的前向扫描器）则把两端点都包含进去。`HemispherePattern.num_rays()` 严格返回
`num_rays_target`——Fibonacci 格点不会产生取整误差。

### TerrainHeightSensor

锚定在某个 link 上的 2D 下射光线网格，输出每条光线相对地形的高度（正值表示在地形上方），
适合作为 critic 的 privileged `height_scan` 观测。默认 backend 把每条光线打到位于
`ground_height` 的水平面上；当场景挂上 `TerrainImporter` 时，内部 `RayCastSensor` 会
对 height-field 做双线性采样。非平地场景的扩展点是继承 `RayCastSensor` 并重载
`_intersect_world_rays`，可换成 BVH 或其他自定义 backend。

| 字段 | 类型 | 含义 |
|------|------|------|
| `link_name` | `str` | 网格原点锚定的 link。 |
| `pattern` | `GridPattern \| RingPattern \| HemispherePattern` | Pattern 几何。 |
| `attach_yaw_only` | `bool` | 仅按 yaw 旋转 pattern，使其保持水平。 |
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

把 cfg 加入 `InteractiveSceneCfg.sensors` 后，传感器即可通过 `env.sensors[name].data` 访问。

## See also

- [配置系统](configs.md)
- [API 参考](../api/reference.md)
