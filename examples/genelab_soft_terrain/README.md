# genelab-soft-terrain

Legged locomotion over **analytic deformable (soft) terrain** for GeneLab — the stage-0
capstone of [ADR-0001](../../plans/adr/0001-stateful-deformable-terrain.md).

## What it is

A Unitree Go1 standing on a *virtual* soft surface. Instead of rigid contact, each foot is
supported by an analytic compliance force `F_z = max(0, k·d + c·ḋ)` injected at the foot
link (`genelab.terrains.DeformableTerrainCfg` / `DeformableTerrainDriver`), where `d` is the
foot's penetration below the surface. There is **no rigid floor under the feet** — the
scene's default ground plane sits at `z = 0`, well below, purely as a fall backstop.

At equilibrium the four feet sink until the total support equals the robot's weight
(`Σ k·dᵢ ≈ m·g`); a stiffer `k` gives shallower sinkage. The simulator's true per-foot
sinkage is exposed as a **privileged observation** (`mdp.terrain_sinkage`) for future
teacher / terrain-identification consumers.

## Traction and walking

The normal force settles the feet at their sinkage equilibrium; the stage-1 `μ` traction
term (`coulomb_tangential_force`, capped optionally by a granular shear strength `η`) gives
the feet grip so they hold position instead of drifting. With **zero actions** this is a
*stand-and-hold* demo. Actually **walking** on the soft terrain needs a trained policy —
this env is the substrate for that next step.

## Run

```python
from genelab_soft_terrain import go1_soft_stand_env_cfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

env = ManagerBasedRlEnv(go1_soft_stand_env_cfg(play=True))
```

The extension also registers the env as `go1-soft-stand-env`
(`genelab_soft_terrain.tasks:register`).
