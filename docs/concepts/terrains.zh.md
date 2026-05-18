# 地形

GeneLab 地形是生成式 height-field 网格，可导入 Genesis scene，并被 ray-based sensor 和 curriculum 读取。

## 为什么需要地形层

locomotion 任务需要地形几何、spawn origin、难度等级和 height scan 保持一致。把地形生成放在一层里，可以让 scene、curriculum 和 sensor 共享同一事实来源。

## 构件

| 配置 | 形态 |
|---|---|
| `FlatPatchCfg` | 平坦子地形。 |
| `PyramidStairsCfg` | 楼梯 pattern。 |
| `RandomRoughCfg` | 随机粗糙表面。 |
| `SlopeCfg` | 斜坡 patch。 |
| `WaveCfg` | 波浪状 height field。 |
| `TerrainGeneratorCfg` | 网格布局与子地形选择。 |
| `TerrainImporter` | 到 Genesis 的运行时桥接。 |

## 接入点

| 消费者 | 用途 |
|---|---|
| `InteractiveScene` | 把生成几何加入 Genesis scene。 |
| `TerrainHeightSensor` | 在 ray grid 下采样高度。 |
| `terrain_levels_vel` curriculum | 按表现提升/降低 env 难度。 |

## 继续阅读

- [传感器](sensors.md)
- [Manager 与 MDP term](managers.md)
- [API 参考](../api/reference.md)
