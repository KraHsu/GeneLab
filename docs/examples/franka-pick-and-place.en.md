# Franka Pick-and-Place

Goal-conditioned manipulation task for the asset-zoo Franka Panda, trained
with Stable-Baselines3 SAC + Hindsight Experience Replay. A 4 cm cube spawns
in front of the robot and each environment samples a target either on the
ground plane or in the air (panda-gym `PandaPickAndPlace` distribution).

## Task

| Task id | Action dim | Algorithm |
|---|---:|---|
| `GeneLab-Franka-Pick-And-Place-v0` | 4 | SB3 SAC + HER + lift bonus + FSM demo prefill |

The action vector is `(dx, dy, dz, gripper)` — `DifferentialIKAction` (with
orientation locked to the panda-gym downward pose) on the seven arm joints
plus `ContinuousGripperAction` on the fingers.

## Installing the extension

```bash
uv pip install -e examples/franka
genelab list tasks | grep Franka
```

Without installation:

```bash
PYTHONPATH=examples/franka/src \
  genelab --import genelab_franka.tasks list tasks
```

The first run downloads the Franka MJCF asset and builds the Genesis kernel
cache.

## Smoke run

```bash
genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 16 --max-iterations 2000
```

## Full training run

The HER policy plateaus around 30 % success without help — sparse goal reward
alone gives no learning signal in the air region of the goal space. Two
ingredients break the plateau and they work in tandem:

1. **Lift bonus** in the env reward (`mdp.lift_bonus`, weight `0.2`): a
   per-step ramp from `0` (cube on the table) to `+0.2` (cube 10 cm above
   the table), mirrored exactly in the HER compute-reward callback so the
   in-buffer and relabelled rewards share one shape.
2. **FSM demo prefill**: a hand-crafted controller in `demo_fsm.py`
   produces full `reach → grasp → lift → place` trajectories, which
   `collect_demos.py` saves as `.npz`. The SB3 backend reads
   `GENELAB_SB3_DEMO_PATH` (or `Sb3AgentCfg.demo_path`) and replays the
   transitions into the replay buffer before `model.learn` starts.

```bash
# 1. Collect demos (about a minute at num-envs 32).
python -m genelab_franka.collect_demos \
  --num-envs 32 --steps 6400 --out /tmp/franka_pp_demos.npz

# 2. Train with demo prefill.
GENELAB_SB3_DEMO_PATH=/tmp/franka_pp_demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
```

Expected milestones on the success-rate curve:

| step bucket | mean success | what's happening |
|---|---|---|
| 0–200 K     | 5–7 %   | demos sit in buffer, SAC has not internalised them yet |
| 200 K–400 K | 10–20 % | policy starts mimicking the lift |
| 400 K–600 K | 30–40 % | crosses the table-only ceiling |
| 600 K–1 M   | 60–85 % | task essentially solved |
| 1.4 M+      | 95 %+   | converged plateau, peak ≈ 99 % |

## Replaying a checkpoint

```bash
genelab play GeneLab-Franka-Pick-And-Place-v0 \
  --checkpoint logs/sb3/franka_pick_and_place/<run>/model.zip \
  --steps 200
```

## Physics overrides

The env config tightens three of the asset-zoo Franka defaults, scoped to
this task only:

| Component | Default | Override | Why |
|---|---|---|---|
| `panda_arm.stiffness` / `damping` | 400 / 80 | **2000 / 200** | Stock PD sags below recoverable z once carrying the cube; stiffer PD holds the arm against gravity + cube load. |
| `panda_hand.effort_limit` / `velocity_limit` | 20 / 0.2 | **100 / 1.0** | Defaults close the fingers at ~ 0.1 rad/s — slower than the cube takes to fall out of the gripper opening. New limits shut the fingers in roughly one env step. |
| Cube `friction` | Genesis default | **1.0** | Genesis's built-in rigid friction is too low to wedge the cube between parallel fingers. Matches panda-gym's MuJoCo cube. |

## Code entry points

| File | Role |
|---|---|
| `tasks.py` | Registers the robot, env, and task id. |
| `env_cfg.py` | Builds the scene, observation groups, reward terms, and physics overrides. |
| `mdp.py` | Task-specific observations, rewards, terminations, reset events. |
| `sb3_cfg.py` | SAC + HER agent config and the HER `compute_reward` callback. |
| `demo_fsm.py` | Hand-crafted FSM that produces `reach → grasp → lift → place` demos. |
| `collect_demos.py` | CLI that runs the FSM through the env and saves transitions to `.npz`. |
| `robot.py` | Wraps the asset-zoo Franka with the panda-gym neutral pose and the Jacobian/IK toggle. |

## Notes

- The cube rests on the ground plane; there is no separate table mesh.
- Goal `z` is sampled uniformly in `[0, 0.2]` with probability `0.7`;
  otherwise the goal is placed at cube height on the ground plane.
- The success distance threshold is `0.05 m`.

## See also

- [MDP terms reference](../concepts/mdp.md)
- [Task design](../best-practices/task-design.md)
- [Asset zoo](../concepts/asset_zoo.md)
