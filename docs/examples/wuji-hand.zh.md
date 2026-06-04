# 舞肌手

舞肌手示例是 play-only 灵巧手 playback task，展示资产打包和脚本化关节轨迹回放。

## 任务

```text
GeneLab-Wuji-Hand-Playback-v0
```

## 运行

```bash
uv pip install -e examples/wuji
genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --steps 500
```

常用 override：

```bash
genelab play GeneLab-Wuji-Hand-Playback-v0 --env.reset_interval 0
genelab play GeneLab-Wuji-Hand-Playback-v0 --env.robot.side left
```

## 展示内容

- 灵巧手 articulation 资产打包。
- 不带 RL runner 的 playback task 结构。
- 可配置 reset/playback 行为。

## 另见

- [场景与实体](../concepts/scene.md)
- [资产库](../concepts/asset_zoo.md)
