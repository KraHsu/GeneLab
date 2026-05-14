# 地形

`genelab.terrains` 是 Genesis 原生 `gs.morphs.Terrain` morph 之上的一层 GeneLab 风格薄包装。子地形 cfg 对应 Genesis 内建的 subterrain 字符串类型，`TerrainGenerator` 把它们拼成一个二维网格，`TerrainImporter` 负责把地形 spawn 进场景并跟踪每个 env 的课程状态。

## 为什么需要专门的一层

Genesis 自带 height-field 地形原语，但其原生 API 使用按字符串名键控的 pydantic options（`"pyramid_stairs_terrain"`、`"random_uniform_terrain"`）。GeneLab 的 cfg 层把接口保持成 dataclass 形态（与项目其它部分一致），让每种子地形拥有带显式字段的 typed Python 类，并跟踪每个 env 的 spawn / level 状态，让课程项可以提升和降级而不直接接触 Genesis 句柄。

## 自带 5 种子地形

| 类 | Genesis 类型 | 参数 |
|---|---|---|
| `FlatPatchCfg` | `flat_terrain` | 无 |
| `PyramidStairsCfg` | `pyramid_stairs_terrain` | `step_width`、`step_height` |
| `RandomRoughCfg` | `random_uniform_terrain` | `min_height`、`max_height`、`step`、`downsampled_scale` |
| `SlopeCfg` | `sloped_terrain` | `slope` |
| `WaveCfg` | `wave_terrain` | `num_waves`、`amplitude` |

`PyramidStairsCfg(step_height=-0.1)` 生成一圈圈向下的同心方形台阶；正的 `step_height` 会反转金字塔方向。`RandomRoughCfg` 在 `downsampled_scale` 下采样后再上采样，所以表观特征尺寸与 `horizontal_scale` 无关。`SlopeCfg(slope=-0.5)` 把整块斜成一个固定坡（负号沿 Genesis 默认的下坡方向）；`WaveCfg` 在整块上铺正弦起伏 —— 介于平地与金字塔台阶之间的中间难度。Genesis 还有另外 4 种类型（`pyramid_sloped`、`discrete_obstacles`、`stairs`、`stepping_stones`）尚未包装 —— 扩展 `SubTerrainCfg` 暴露它们是一个 20 行的小活。

## 拼装地形网格

```python
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    TerrainGeneratorCfg,
)
from genelab.configs import InteractiveSceneCfg

scene = InteractiveSceneCfg(
    env_spacing=(0.0, 0.0),
    terrain=TerrainGeneratorCfg(
        num_rows=4,
        num_cols=4,
        subterrain_size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        sub_terrains={
            "flat": FlatPatchCfg(proportion=1.0),
            "stairs": PyramidStairsCfg(step_width=0.5, step_height=-0.08, proportion=2.0),
            "rough": RandomRoughCfg(min_height=-0.05, max_height=0.05, proportion=1.0),
        },
        curriculum=True,
        seed=0,
    ),
)
```

`sub_terrains` 是 `dict[str, SubTerrainCfg]`；key 是本地 cfg 标签，value 携带 Genesis 参数。`layout` 默认为 `None`（按 `proportion` 随机平铺）；传入 `tuple[tuple[str, ...], ...]` 形态的 cfg key 二维表则可以确定性地固定网格。

!!! warning "每个 Genesis 类型只有一套参数"
    Genesis 的 `subterrain_parameters` 按*类型字符串*键控，而不是按单元格。两个 `step_width` 不同的 `PyramidStairsCfg` 实例会塌成单一一套参数。要区分几何就用不同的 key，要做同类型内变化就在 importer 上启用 `randomize=True`。

## ray-cast 传感器接入

当 `InteractiveSceneCfg.terrain` 被设置时，`InteractiveScene.build()` 会 spawn `gs.morphs.Terrain` 而不是默认 plane，并把 importer 暴露为 `scene.terrain`。`RayCastSensor` 检测到地形后，会对每条射线在世界坐标 `(x, y)` 处对 `terrain.heightfield_tensor` 做双线性采样，得到命中点高度。`TerrainHeightSensor` 是同一路径上的薄封装，所以每个关节的 height scan 输出反映的是真实地形而不是常数平面。

垂直 / 接近垂直的射线，输出与底层 height field 的误差不超过 `vertical_scale`。斜射线使用射线起点 `(x, y)` 处的高度 —— 对 height-scan 网格足够准，但对 BVH 风格查询是近似的；这类场景请覆写 `RayCastSensor._intersect_world_rays`。

## 课程项: `terrain_levels_vel`

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"distance_threshold": 2.0, "demote_ratio": 0.5},
    ),
}
```

每次 reset，该项把每个 env 的已走距离（`root_pos - spawn_pos` 在 XY 上的范数）与 `distance_threshold` 比较。走得超过阈值的 env 在难度网格中上移一行（封顶在 `num_rows - 1`）；走得不到 `distance_threshold * demote_ratio` 的 env 下移一行（地板在 0）。课程项随后通过 `Articulation.write_root_state` 把新 spawn 位姿写入仿真，下一个 episode 就从新子地形开始。manager 自动以 `Curriculum/terrain_levels` 记录平均 level。

## 需要知道的失败模式

* **`sub_terrains` 为空** —— `TerrainGenerator.__init__` 立刻抛错；生成器没东西可拼。
* **`layout` 形状不匹配** —— `ValueError` 列出实际形状与声明形状。
* **`layout` 引用未知 key** —— `ValueError` 列出缺失的 key。
* **build 前访问 per-env 状态** —— `terrain_levels` / `terrain_cols` / `spawn_pos` 在 `init_per_env_state` 之前抛 `RuntimeError`；`InteractiveScene.build` 会自动调一次。
* **spawn 前访问 `heightfield`** —— `RuntimeError`；先 spawn 再读。

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
- [Actuators](actuators.md)
