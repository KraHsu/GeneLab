# Franka Pick-and-Place

Goal-conditioned pick-and-place task for the Franka Panda (asset zoo `franka`),
trained through Stable-Baselines3 SAC + Hindsight Experience Replay. A 4 cm
cube spawns in front of the arm and the policy must move it to a per-env target
sampled either on the table or in the air (panda-gym `PandaPickAndPlace`
distribution).

The setup combines four pieces, each load-bearing:

1. **4-DoF Cartesian action** — `DifferentialIKAction(body_name="hand")`
   (orientation locked to the panda-gym downward pose) on the seven arm joints
   plus `ContinuousGripperAction` on the fingers. Action vector is
   `(dx, dy, dz, gripper)`.
2. **SAC + HER with sparse goal reward + lift bonus** — `-1` outside a
   `0.05 m` threshold to the goal, `0` inside, plus a per-step ramp from `0`
   (cube on table) to `+0.2` (cube 10 cm above table). The lift term is the
   only thing that escapes the "scoot on the table" plateau HER falls into
   when relabelling has nothing to relabel into.
3. **Physics overrides** — stiffer `panda_arm` PD so the loaded arm can lift,
   faster `panda_hand` actuator so a single `-1` gripper action closes in one
   env step, and `friction=1.0` on the cube so it stays wedged between the
   fingers under lift acceleration.
4. **FSM demo prefill** — a hand-crafted controller in `demo_fsm.py` produces
   full `reach -> grasp -> lift -> place` trajectories, which
   `collect_demos.py` saves as `.npz`; the SB3 backend reads
   `GENELAB_SB3_DEMO_PATH` (or `Sb3AgentCfg.demo_path`) and pre-fills the
   replay buffer before `model.learn`. Without demos, training plateaus
   around 30% success; with them it reaches 95%+ inside 1.5 M steps.

## Install

```bash
uv pip install -e examples/franka
```

## Smoke run (no convergence)

```bash
uv run genelab list tasks | grep Franka-Pick-And-Place
uv run genelab train GeneLab-Franka-Pick-And-Place-v0 --gpu --num-envs 16 --max-iterations 2000
```

The first run downloads the Franka MJCF and Genesis kernel cache; subsequent
runs start instantly. Add `--vis` to `genelab play` to open the viewer.

## Full training (convergence)

```bash
# 1. Collect FSM demos (about a minute at num-envs 32).
uv run python -m genelab_franka.collect_demos \
    --num-envs 32 --steps 6400 --out /tmp/franka_pp_demos.npz

# 2. Train with demo prefill (~50 minutes on a single H200).
GENELAB_SB3_DEMO_PATH=/tmp/franka_pp_demos.npz \
    uv run genelab train GeneLab-Franka-Pick-And-Place-v0 \
    --gpu --num-envs 32 --max-iterations 2000000
```

Expected milestones on the success-rate curve (with demo prefill):

| step bucket | mean success | what's happening |
|-------------|--------------|------------------|
| 0–200 K     | 5–7 %        | demos sit in buffer, SAC hasn't internalised them yet |
| 200 K–400 K | 10–20 %      | policy starts mimicking the lift |
| 400 K–600 K | 30–40 %      | crosses the table-only ceiling |
| 600 K–1 M   | 60–85 %      | task essentially solved |
| 1.4 M+      | 95 %+        | converged plateau, peak ≈ 99 % |

## Notes

- The scene has no table — the cube rests on the ground plane. Robot base is
  at `(0, 0, 0)`; panda-gym uses `(-0.6, 0, 0)` with a table.
- Goal `z` is sampled uniformly in `[0, 0.2]` with probability `0.7`,
  otherwise forced to `cube_size/2` (panda-gym's "on the table" branch).
- The actuator/friction overrides are scoped to this env config; the asset-zoo
  `franka` defaults remain Isaac Lab's stock values for other tasks that pull
  the same robot.
