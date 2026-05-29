# Eval and Export

GeneLab's research-reproducibility tooling under
`genelab.rl.evaluator` / `eval_callback` / `exporter`, surfaced as three
CLIs that close the **train → eval → export** loop:

| Command | Purpose | Output |
|---|---|---|
| `genelab eval TASK CKPT` | Deterministic rollout, fixed seed, N episodes | `eval.json` |
| `genelab train ... --eval-every K` | Periodic in-training eval + best-model save | `logs/.../best_model.<ext>` + `best_model_meta.json` |
| `genelab export TASK CKPT` | Backend-agnostic TorchScript / ONNX policy | `policy.{ts,onnx}` + `<file>.metadata.json` |

All three route through the same backend abstraction (`InferenceSetup`, defined
in `genelab.rl.backends.base`), so they work identically against the `rsl_rl`,
`skrl`, and `sb3` backends.

## `genelab eval`

Runs a vectorized deterministic rollout and writes a JSON summary with this
schema:

```bash
genelab eval GeneLab-Inverted-Pendulum-v0 logs/rsl_rl/exp1/.../model_500.pt \
    --num-envs 64 --episodes 100 --seed 0 \
    --deterministic --out eval.json
```

Output:

```json
{
  "task": "GeneLab-Inverted-Pendulum-v0",
  "checkpoint": "logs/.../model_500.pt",
  "num_episodes": 100,
  "metrics": {
    "return_mean": 487.3,
    "return_std": 22.1,
    "length_mean": 998.4,
    "success_rate": 0.96
  },
  "wall_clock_seconds": 18.2,
  "seed": 0,
  "deterministic": true,
  "evaluated_at": "2026-05-20T08:42:11+00:00"
}
```

### Success rate

`success_rate` is computed when the task publishes a per-env bool tensor at
`extras["is_success"]` from `ManagerBasedRlEnv.step` (gymnasium convention).
Tasks opt in by setting `self._extras["is_success"] = <(num_envs,) bool tensor>`
inside a termination or reward term — typically a check against the goal pose
for manipulation or a "reached velocity command" check for locomotion.

Tasks that do **not** publish `is_success` get `success_rate: null` in the
output; downstream tools (best-model selection, reference-runs tables) should
guard against `None`.

## `genelab train --eval-every`

When `--eval-every K` is set, training runs in chunks of `K` iterations. After
each chunk the latest checkpoint is loaded into the same backend and a
deterministic eval is run (defaulting to 10 episodes at the same `num_envs` as
training). When `return_mean` improves on the prior best, the checkpoint is
copied to `<log_dir>/best_model.<ext>` and a sibling `best_model_meta.json` is
updated with the eval payload.

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
    --max_iterations 1000 --num_envs 64 --seed 0 \
    --eval-every 100 --eval-episodes 16
```

Caveats:

- Each chunk closes and rebuilds the Genesis env via the backend's normal train
  lifecycle. Pick `--eval-every` ≥ 50 for short tasks so Genesis init time is
  amortized.
- For off-policy algorithms (SAC / TD3 / DDPG via `skrl` or `sb3`), reloading
  from a checkpoint between chunks loses the replay buffer. Sample efficiency
  degrades but training still converges.
- `best_model.<ext>` reuses the source backend's checkpoint format (`.pt` for
  `rsl_rl` / `skrl`, `.zip` for `sb3`). The metadata file records the source
  iteration, eval seed, episodes, and return statistics.

## `genelab export`

Serializes the actor sub-network to **TorchScript** or **ONNX** with per-term
obs `scale` / `clip` baked into a single `forward(raw_obs) -> actions` pass.
Deployment environments need only `torch` (TorchScript) or an ONNX runtime;
they do **not** need `rsl_rl` / `skrl` / `stable_baselines3` at inference time.

```bash
# TorchScript
genelab export Genelab-Velocity-Flat-Unitree-G1-v0 logs/.../model_30000.pt \
    --format torchscript --out policy.ts

# ONNX (opset 17 by default)
genelab export Genelab-Velocity-Flat-Unitree-G1-v0 logs/.../model_30000.pt \
    --format onnx --out policy.onnx --opset 17
```

> **Note**: `GeneLab-Franka-Pick-And-Place-v0` is SAC+HER with a goal-conditioned
> `Dict` observation. Its exported model takes a single flat `obs` that is the
> concatenation of `observation` + `achieved_goal` + `desired_goal` (in that
> order); see the multi-group metadata below. Locomotion tasks (cartpole, G1) use
> a single flat-tensor obs group.

The exporter writes a sibling `<output>.metadata.json` describing the obs
schema. `obs_dim` is the total flat-input width; each `obs_groups` entry records
its `start` offset into that flat tensor (so goal-conditioned policies can be
sliced back into their sub-spaces):

```json
{
  "task": "Genelab-Velocity-Flat-Unitree-G1-v0",
  "checkpoint": "logs/.../model_30000.pt",
  "obs_dim": 48,
  "obs_groups": {
    "policy": {
      "start": 0,
      "dim": 48,
      "terms": [
        {"name": "joint_pos", "dim": 23, "start": 0, "scale": 1.0, "clip": null},
        {"name": "joint_vel", "dim": 23, "start": 23, "scale": 0.1, "clip": [-2, 2]}
      ]
    }
  },
  "action_dim": 23,
  "action_range": [-1.0, 1.0],
  "normalization_baked": true,
  "format": "torchscript",
  "exported_at": "2026-05-20T08:42:11+00:00",
  "torch_version": "2.4.0"
}
```

For a SAC+HER task `obs_groups` has three entries — e.g. `observation`
(`start: 0`), `achieved_goal` (`start: 35`), `desired_goal` (`start: 38`) — and
`obs_dim` is their sum.

### Deployment-side usage

```python
import torch
m = torch.jit.load("policy.ts")
m.eval()
# raw obs in (training-side concatenation order); model applies scale/clip itself
actions = m(torch.tensor([[joint_pos_0, joint_pos_1, ..., joint_vel_0, ...]]))
```

For ONNX:

```python
import onnxruntime as ort
sess = ort.InferenceSession("policy.onnx")
actions = sess.run(None, {"obs": raw_obs.astype("float32")})[0]
```

### What's exported

The actor is extracted via a backend-specific shim and wrapped so the call
shape is uniform:

- `rsl_rl`: takes the actor module off the algorithm directly
  (`alg._raw_actor`, falling back to `alg.actor`) and uses its `as_jit()` export
  wrapper, which exposes a flat `forward(obs) -> deterministic action` with the
  learned obs normalizer baked in. Older releases that kept the actor under
  `alg.actor_critic.actor` (or only `act_inference`) are still supported.
- `skrl`: wraps `agent.policy.act` and returns the deterministic mean (the
  `mean_actions` key) for `GaussianMixin` policies.
- `sb3`: wraps `model.policy._predict(obs, deterministic=True)`, which is
  uniform across PPO / A2C / SAC / TD3 / DDPG. For goal-conditioned **SAC+HER**
  policies the observation space is a `Dict`
  (`observation` / `achieved_goal` / `desired_goal`) and the SAC actor consumes
  all keys, so the wrapper takes the flat concatenation of those sub-spaces (in
  that order) and rebuilds the Dict before calling the policy — the exported
  model still has a single flat `obs` input, and the metadata's `obs_groups`
  records each sub-space's `start` / `dim` so deployers know the layout.

> SB3's ONNX export uses the legacy TorchScript-based exporter
> (`torch.onnx.export(..., dynamo=False)`): the `torch.export`-based default
> (torch ≥ 2.9) can't trace SAC's `Normal` distribution construction.

### Limitations

- Recurrent policies (rsl_rl `rnn_type` set) are not yet supported — the
  exported model has no hidden-state slot.
- The exported model does **not** apply observation noise from `ObservationTermCfg.noise`;
  noise is part of training only.
