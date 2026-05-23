# Contracts and Ports

`genelab.contracts` is the small public boundary that domain code type-hints against. It keeps
managers, MDP terms, and sensors coupled to the env/scene **surface** they need, not to the concrete
Genesis-backed implementation that happens to provide it today.

## Why this band exists

Most task logic only needs a few stable attributes: device, number of envs, the scene, managers,
entity tables, and state-writing helpers. Importing `ManagerBasedRlEnv` or `InteractiveScene`
directly from every term would point the domain layer back at the runtime layer and make a second
env implementation harder to add.

The contracts invert that dependency:

```text
domain term -> EnvContext / SceneContext protocol
concrete env -> implements that protocol
```

This is a type boundary, not a wrapper. Runtime objects are still the normal env and scene objects;
the protocols only document the attributes that domain code is allowed to rely on.

## EnvContext

`EnvContext` is the term-facing environment surface. Use it when an observation, reward,
termination, event, curriculum, metric, action, or command function needs the env:

```python
from genelab.contracts import EnvContext


def base_height(env: EnvContext):
    robot = env.articulations["robot"]
    return robot.data.root_pos_w[:, 2]
```

The protocol includes read-only runtime facts such as `device`, `num_envs`, `dt`, episode length
buffers, `scene`, `articulations`, and `sensors`, plus the manager attributes and state-writing
helpers that domain terms use.

## SceneContext

`SceneContext` is the surface reached through `env.scene`. Use it when code only needs scene-level
state such as `num_envs`, `device`, `env_origins`, Genesis scene access, terrain, sensors,
articulations, rigid objects, recorder bridge, or viewer state.

Prefer `EnvContext` for normal MDP terms because it keeps access to the managers and write helpers.
Use `SceneContext` for helpers that are truly scene-only.

## NoiseCfg

`NoiseCfg` now lives in `genelab.contracts` because the observation manager needs to type-hint
noise without importing `genelab.mdp`. Concrete noise models such as `Unoise`, `Gnoise`,
`ScaledNoise`, `CorrelatedNoise`, and `BiasDrift` still live in `genelab.mdp.noise` and remain
available through the usual MDP and facade imports.

## Where to continue

- [Module Map](../reference/module-map.md)
- [Managers and MDP terms](managers.md)
- [MDP Terms Reference](mdp.md)
