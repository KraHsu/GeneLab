# Wuji Hand

The Wuji Hand example is a play-only articulated-hand playback task. It demonstrates asset packaging
and scripted joint trajectory playback.

## Task

```text
GeneLab-Wuji-Hand-Playback-v0
```

## Running

```bash
uv pip install -e examples/wuji
genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --steps 500
```

Useful overrides:

```bash
genelab play GeneLab-Wuji-Hand-Playback-v0 --env.reset_interval 0
genelab play GeneLab-Wuji-Hand-Playback-v0 --env.robot.side left
```

## Shows

- Articulated hand asset packaging.
- Playback task structure without an RL runner.
- Configurable reset/playback behavior.

## See also

- [Scene and entities](../concepts/scene.md)
- [Asset zoo](../concepts/asset_zoo.md)
