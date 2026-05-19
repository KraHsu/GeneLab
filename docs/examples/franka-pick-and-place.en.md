# Franka Pick-and-Place

The Franka pick-and-place example shows a goal-conditioned manipulation task with the asset-zoo
Franka Panda arm. A 4 cm cube spawns in front of the robot, and each environment samples a target
position either on the ground plane or in the air. The reward follows the dense panda-gym shape:
end-effector-to-cube distance, cube-to-goal distance, and a small success bonus inside the goal
threshold.

## Tasks

| Task id | Action dim | Shows |
|---|---:|---|
| `GeneLab-Franka-Pick-And-Place-v0` | 9 | Raw joint-position control for the 7 arm joints and 2 finger joints. |
| `GeneLab-Franka-Pick-And-Place-Cartesian-v0` | 4 | Panda-gym-style `(dx, dy, dz, gripper)` control via differential IK and a binary gripper. |

Use the joint-position task for ablations against direct joint control. Use the Cartesian task when
you want the same 4-DoF action surface as panda-gym `PandaPickAndPlace`.

## Install and list

```bash
uv pip install -e examples/franka_pick_and_place
uv run genelab list tasks | grep Franka
```

Without installation:

```bash
PYTHONPATH=examples/franka_pick_and_place/src \
  uv run genelab --import genelab_franka_pick_and_place.tasks list tasks
```

The first run may download the Franka MJCF asset and build the Genesis kernel cache.

## Run a smoke training job

Joint-position variant:

```bash
uv run genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --num-envs 16 \
  --max-iterations 2
```

Cartesian variant:

```bash
uv run genelab train GeneLab-Franka-Pick-And-Place-Cartesian-v0 \
  --num-envs 16 \
  --max-iterations 2
```

Add `--vis` to open the viewer for a small visual run.

## Replay a checkpoint

```bash
uv run genelab play GeneLab-Franka-Pick-And-Place-Cartesian-v0 \
  --checkpoint logs/rsl_rl/franka_pick_and_place/<run>/model_2.pt \
  --steps 50
```

Use the matching task id for the checkpoint you trained. The PPO runner config is shared by both
action variants, but the policy network input/output shape depends on the task's action space.

## Action variants

| Variant | Action terms | Policy output |
|---|---|---|
| Joint position | `JointPositionActionCfg(joint_names=(arm, fingers))` | 9 joint-position targets with the robot default pose as offset. |
| Cartesian | `DifferentialIKActionCfg(body_name="hand", joint_names=(arm,))` + `BinaryGripperActionCfg(joint_names=(fingers,))` | 3 end-effector position deltas plus one gripper scalar. |

The Cartesian variant enables `requires_jac_and_ik=True` on the Franka articulation so Genesis can
provide the end-effector Jacobian. `DifferentialIKAction` solves a damped-least-squares IK step each
control tick and writes partial arm joint targets. `BinaryGripperAction` maps a single scalar to
`closed_pos=0.0` or `open_pos=0.04` for both finger joints.

## Code entry points

| File | Role |
|---|---|
| `tasks.py` | Registers the robot, envs, and both task ids. |
| `env_cfg.py` | Builds the shared scene and selects the joint-position or Cartesian action config. |
| `mdp.py` | Defines task-specific observations, rewards, reset sampling, and terminations. |
| `robot.py` | Wraps the asset-zoo Franka and toggles the Jacobian/IK requirement for Cartesian control. |

## Notes

- The cube rests on the ground plane; there is no separate table mesh in the scene.
- The robot base starts at `(0, 0, 0)`, unlike panda-gym's table setup.
- Goal `z` is sampled uniformly in `[0, 0.2]` with probability `0.7`; otherwise the goal is placed
  at cube height on the ground plane.
- The success distance threshold is `0.05 m`.

## See also

- [MDP terms reference](../concepts/mdp.md)
- [Task design](../best-practices/task-design.md)
- [Asset zoo](../concepts/asset_zoo.md)
