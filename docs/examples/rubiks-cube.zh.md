# 魔方

魔方示例是 play-only scene，展示刚体组合和可视交互。

## 任务

```text
GeneLab-Rubiks-Play-v0
```

## 运行

```bash
uv pip install -e examples/genelab_examples
uv run genelab play GeneLab-Rubiks-Play-v0 --vis --steps 500
```

常用 override：

```bash
uv run genelab play GeneLab-Rubiks-Play-v0 --env.robot.cubie_size 0.04
uv run genelab play GeneLab-Rubiks-Play-v0 --env.robot.welded true
```

## 展示内容

- 非 RL play task 注册。
- 多刚体 scene 组合。
- 可视参数配置 override。

## 另见

- [场景与实体](../concepts/scene.md)
- [配置参考](../reference/configuration.md)
