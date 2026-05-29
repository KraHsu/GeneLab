# Rubik's Cube

The Rubik's Cube example is a play-only scene that demonstrates rigid-object composition and visual
interaction.

## Task

```text
GeneLab-Rubiks-Play-v0
```

## Running

```bash
uv pip install -e examples/genelab_examples
genelab play GeneLab-Rubiks-Play-v0 --vis --steps 500
```

Useful overrides:

```bash
genelab play GeneLab-Rubiks-Play-v0 --env.robot.cubie_size 0.04
genelab play GeneLab-Rubiks-Play-v0 --env.robot.welded true
```

## Shows

- Non-RL play task registration.
- Scene composition from many rigid bodies.
- Config overrides for visual parameters.

## See also

- [Scene and entities](../concepts/scene.md)
- [Configuration Reference](../reference/configuration.md)
