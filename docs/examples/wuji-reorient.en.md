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
- **Curriculum** (training only) — a success curriculum tightens the goal tolerance from
  loose (0.8 rad) to the target (0.2 rad) as the policy reliably reaches goals, and an
  adaptive-episode curriculum ramps the cube velocity disturbance with episode survival.

## Domain randomization

Training randomizes hand friction, link mass / COM, PD gains, and encoder bias, plus a
periodic cube velocity disturbance; evaluation (`--play`) runs nominal physics with these
stripped.

!!! note "Genesis vs MuJoCo contact DR"
    The mjlab reference also randomizes MuJoCo-specific `sol_params` (soft-pad compliance),
    geom size, and inertia tensors. Those have no Genesis equivalent (different contact
    solver; no per-env geom resizing or inertia setter), so they are omitted here.

## Convergence

Reference-scale run (8192 envs, 5000 iterations, RTX 5060 Ti, ~5 h) with the curriculum:

- The success curriculum tightens the tolerance to the target 0.2 rad by ~iter 1000, after
  which the policy keeps improving *at full difficulty* (~6.7 goals reached per episode by
  the end, with a stable grip — `cage_drop` ≈ 0.2).
- Deterministic eval over 100 episodes (`genelab eval`, tight 0.2 threshold): **success rate
  0.99** (fraction of episodes that reorient the cube to at least one *held* SO(3) goal),
  mean return ~1060, mean episode length ~591.

!!! note "The curriculum is essential"
    Without the success curriculum, the mjlab-aligned reward weights (heavy torque /
    action-rate / cage penalties) over-regularize into a static hold and reach **0%** at a
    short budget; with the curriculum they reach the release-level 0.99. A short run without
    the curriculum and with lighter Genesis-tuned weights reaches ~0.33.

## See also

- [Wuji Hand](wuji-hand.md)
- [Asset zoo](../concepts/asset_zoo.md)
