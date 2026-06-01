# Wuji Hand Reorientation

SO(3) in-hand cube reorientation for the WUJI Hand: a fixed-base, palm-up dexterous hand
must rotate a free cube to a stream of random orientation goals (expressed in the wrist
"tag" frame) and hold each within a tolerance window without dropping it. The task is a
Genesis-adapted port of the mjlab `reorient` reference, trained with RSL-RL PPO.

## Task

```text
Genelab-Reorient-Wuji-Hand-v0
```

20-DoF right hand (5 fingers × 4 joints), a 54 mm cube, and an SO(3) goal command with a
hold-and-advance success cycle.

## Running

```bash
uv pip install -e examples/wuji
genelab train Genelab-Reorient-Wuji-Hand-v0 --num_envs 4096 --gpu
genelab play  Genelab-Reorient-Wuji-Hand-v0 --checkpoint logs/rsl_rl/wuji_reorient/<run>/model.pt --vis
```

## MDP design

- **Action** — joint-position offset with EMA smoothing + startup warmup
  (`JointPositionOffsetEMAAction`), 20-d, scaled around the home grasp keyframe.
- **Command** — `InHandReorientCommand`: samples goals uniformly on SO(3) in the tag frame;
  an APPROACHING → SUCCESS_WINDOW state machine counts in-tolerance steps and advances to a
  new goal after a hold window.
- **Rewards** — orientation alignment (geodesic tolerance), an escalating hold bonus, a
  palm-relative AABB "cage" escape penalty, hand-pose / action-rate / torque regularizers,
  and contact terms (fingertip slide, palm-detach, finger self-collision) driven by a custom
  `get_contacts` hand-cube sensor.
- **Observations** — policy: joint pos/vel, cube position in the tag frame, 6D goal-rotation
  error, last action; critic adds command-state and cage-counter progress.
- **Termination** — time-out, or `cage_drop` when the cube leaves the palm cage long enough.

## Domain randomization

Training randomizes hand friction, link mass / COM, PD gains, and encoder bias, plus a
periodic cube velocity disturbance; evaluation (`--play`) runs nominal physics with these
stripped.

!!! note "Genesis vs MuJoCo contact DR"
    The mjlab reference also randomizes MuJoCo-specific `sol_params` (soft-pad compliance),
    geom size, and inertia tensors. Those have no Genesis equivalent (different contact
    solver; no per-env geom resizing or inertia setter), so they are omitted here.

## Convergence

Short GPU smoke (2048 envs, 400 iterations, RTX 5060 Ti) — a learning-signal check, not a
release-grade policy:

- Orientation-alignment reward rises from ~5 to ~9.4.
- The policy reaches ~3.7 successive SO(3) goals per episode (`goals_reached` metric); most
  episodes survive to time-out rather than dropping the cube.
- Deterministic eval over 100 episodes: mean return ~327, mean episode length ~568 steps.

A full release-scale run (8192 envs, several thousand iterations) is left to the user and
should push the per-goal success rate and goal count higher.

## See also

- [Wuji Hand](wuji-hand.md)
- [Asset zoo](../concepts/asset_zoo.md)
