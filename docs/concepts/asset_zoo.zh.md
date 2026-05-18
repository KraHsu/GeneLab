# 资产库

asset zoo 是 GeneLab 随包提供的一组机器人配置和可下载资产。它适合示例和 smoke test，但下游项目仍应在自己的包里维护机器人配置。

## 随包资产

| 名称 | Factory |
|---|---|
| Unitree G1 | `UnitreeG1Cfg` |
| Unitree Go1 | `UnitreeGo1Cfg` |
| ANYmal C | `AnymalCCfg` |
| Franka Panda | `FrankaPandaCfg` |
| Cartpole | `CartpoleCfg` |
| G1 LAFAN1 motion | `g1_lafan1_dance1_subject2()` |

导入 `genelab.asset_zoo` 会把随包机器人注册进 `ROBOTS`。资产下载保持惰性，只有 factory 真正需要文件时才发生。

## 缓存与校验

下载资产位于项目缓存目录。资产 helper 可以在返回路径前校验 checksum，因此首次获取后，任务能依赖稳定的本地文件。

## 继续阅读

- [模块地图](../reference/module-map.md)
- [Unitree G1](../examples/unitree-g1.md)
- [API 参考](../api/reference.md)
