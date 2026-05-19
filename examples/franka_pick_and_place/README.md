# Franka Pick-and-Place

Goal-conditioned pick-and-place task for the Franka Panda (asset zoo `franka`). A 4 cm
cube spawns in front of the arm and the policy must move it to a per-env target sampled
either on the table or in the air (panda-gym `PandaPickAndPlace` distribution). The
reward shaping mirrors panda-gym's dense form — negative `EE → cube` and `cube → goal`
distances plus a small bonus when the cube is within `0.05 m` of the goal.

## Install

```bash
uv pip install -e examples/franka_pick_and_place
```

## Smoke run (no convergence)

```bash
uv run genelab list tasks | grep Franka-Pick-And-Place
uv run genelab train GeneLab-Franka-Pick-And-Place-v0 --num-envs 16 --max-iterations 2
uv run genelab play GeneLab-Franka-Pick-And-Place-v0 \
    --checkpoint logs/rsl_rl/franka_pick_and_place/<run>/model_2.pt --steps 50
```

The first run downloads the Franka MJCF and Genesis kernel cache; subsequent runs
start instantly. Add `--vis` to either command to open the viewer.

## Notes

- Action space is 9-dim joint position (arm `joint1..7` + `finger_joint1..2`). GeneLab
  ships no EE-delta + IK action term today, so this task uses raw joint-position
  control. panda-gym uses an EE-delta IK action with `action_dim=4`; expect different
  sample-efficiency characteristics.
- The scene has no table — the cube rests on the ground plane. Robot base is at
  `(0, 0, 0)`; panda-gym uses `(-0.6, 0, 0)` with a table.
- Goal `z` is sampled uniformly in `[0, 0.2]` with probability `0.7`, otherwise forced
  to `cube_size/2` (panda-gym's "on the table" branch).
