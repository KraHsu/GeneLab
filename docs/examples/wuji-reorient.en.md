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
  (`JointPositionOffsetEMAAction`), 20-d, scaled around the home grasp keyframe; a small
  per-step action noise is injected during training (see Domain randomization).
- **Command** — `InHandReorientCommand`: samples goals uniformly on SO(3) in the tag frame;
  an APPROACHING → SUCCESS_WINDOW state machine counts in-tolerance steps and advances to a
  new goal after a hold window.
- **Rewards** — orientation alignment (geodesic tolerance), an escalating hold bonus, a
  palm-relative AABB "cage" escape penalty, hand-pose / action-rate / torque regularizers,
  and contact terms (fingertip slide, palm-detach, finger self-collision) driven by a custom
  `get_contacts` hand-cube sensor.
- **Observations** — the actor sees a **3-step history** (term-major, matching the mjlab
  reference) of: joint position relative to the home pose, joint velocity, cube position in
  the tag frame, the 6D cube-to-goal rotation error, and the last action — 69 values per
  step, 207 over the history. The critic adds command-state and cage-counter progress on a
  single step.
- **Termination** — time-out, or `cage_drop` when the cube leaves the palm cage long enough.
- **Curriculum** (training only) — a success curriculum tightens the goal tolerance from
  loose (0.8 rad) to the target (0.2 rad) as the policy reliably reaches goals, and an
  adaptive-episode curriculum ramps the cube velocity disturbance with episode survival.

## Domain randomization

Randomization here targets the *robustness of the closed-loop feedback gait*, not a match to
any single physics parameter — the reason is in [Sim-to-sim transfer](#sim-to-sim-transfer).
Training (stripped at evaluation, which runs nominal physics) applies:

- per-env startup randomization of hand and cube friction, link mass / COM, cube mass, and PD
  gains, plus encoder bias;
- a per-step action noise and a heavier observation noise (joint position / velocity, cube
  position, goal error);
- a frequent linear + angular cube velocity disturbance.

!!! note "Contact-solver randomization"
    Genesis stores geom solver parameters (`sol_params`) globally rather than per-env, so
    per-env contact-compliance randomization — and the MuJoCo-specific geom-size / inertia
    randomizations — have no per-env Genesis equivalent and are omitted. The transfer gap
    turned out not to live in those parameters anyway (see below).

## Convergence

Reference-scale run (8192 envs, 5000 iterations, RTX 5060 Ti, ~5 h):

- The success curriculum tightens the tolerance to the target 0.2 rad by ~iter 1500, after
  which the policy keeps improving *at full difficulty* (~5 goals reached per episode by the
  end, under the active disturbance).
- Deterministic eval over 256 episodes (0.2 threshold, nominal physics): **success rate
  ≈ 1.0** — the fraction of episodes that reorient the cube to at least one *held* SO(3) goal
  — at ~5 goals reached per episode.

The success curriculum is required: its loose→tight tolerance supplies the early reward
signal that lets the heavily-regularized policy learn to reorient rather than just hold.

## Sim-to-sim transfer { #sim-to-sim-transfer }

`sim2sim_mjlab` evaluates the trained policy in the mjlab reference environment itself — its
`scene_builder` (hand + cube + contacts), physics, action pipeline, goal sampling, and drop +
hold/success criterion. Only the policy and an observation/action adapter come from GeneLab.
The two observation layouts differ (mjlab: limit-normalized joint angles + joint-position
target error + tag-frame goal error; GeneLab: joint-position-relative + joint velocity +
world-frame goal error, both 3-step), so the adapter rebuilds the GeneLab actor observation
from the mjlab scene state with the joint-major ↔ finger-major remap and the 3-step history.

Over 100 trials × 3 seeds in the mjlab environment (0.2 threshold): **success rate ≈ 0.65**
(best checkpoint ≈ 0.67) with **drop rate ≈ 0** — the grasp transfers fully; the residual is
goals not *held* within the trial window. Transfer is **non-monotonic in training** and
peaks mid-run (~iter 3500–3800), so the best-transfer checkpoint is not necessarily the last
one. `play_mjlab` drives the same bridge through mjlab's native viewer (full scene + goal
visualization) for inspection. Both run inside the wuji-mjlab environment with GeneLab on
`PYTHONPATH`.

!!! note "Why the randomization matters"
    A policy trained without the robustness randomization holds the cube but barely reorients
    it under mjlab (≈0 success): the regrasping gait overfits to Genesis's exact per-step
    contact response. Sweeping friction, mass, timestep, actuator gains, and the contact
    solver leaves that unchanged, while replaying the in-Genesis winning finger targets does
    move the mjlab cube — so the gap is closed-loop feedback brittleness, which the action /
    observation noise and angular cube disturbances address.

## See also

- [Wuji Hand](wuji-hand.md)
- [Asset zoo](../concepts/asset_zoo.md)
