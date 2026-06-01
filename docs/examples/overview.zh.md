# 示例

示例是扩展包，用来展示 GeneLab 能力，而不把项目专用代码放进 `src/genelab/`。

## 能力地图

| 示例 | 包 | 展示内容 |
|---|---|---|
| [倒立摆](inverted-pendulum.md) | `examples/inverted_pendulum` | 最小 train/play 闭环、manager term、RSL-RL 集成。 |
| [Unitree G1](unitree-g1.md) | `examples/unitree` | 人形机器人 locomotion、速度命令、动作模仿。 |
| [Franka 抓取放置](franka-pick-and-place.md) | `examples/franka` | Goal-conditioned manipulation、SAC + HER + lift bonus + FSM demo prefill。 |
| [Showcase](showcase.md) | `examples/genelab_showcase` | 传感器、ray cast、接触、地形、课程、执行器、录制。 |
| [魔方](rubiks-cube.md) | `examples/genelab_examples` | 刚体组合与可视交互。 |
| [舞肌手](wuji-hand.md) | `examples/genelab_examples` | 灵巧手 playback 与资产打包。 |

## 安装示例

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
uv pip install -e examples/franka
genelab list tasks
```

只有需要 Unitree 时再安装：

```bash
uv pip install -e examples/unitree
```

## 另见

- [教程](../tutorial.md)
- [构建扩展项目](../best-practices/extension-projects.md)
