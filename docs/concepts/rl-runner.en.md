# RL runner

`genelab.rl` is the thin RSL-RL integration that turns a registered task into
a runnable PPO training or replay session. The CLI subcommands `play` and
`train` dispatch into the two entry points covered below; library callers can
also reach them directly.

## Module surface

| Name | Purpose |
|---|---|
| `train_task(task_id, agent_cfg, **kwargs) -> Path` | Build the task's env, wrap it in `RslRlVecEnvWrapper`, and drive RSL-RL's `OnPolicyRunner.learn`. Returns the resolved log directory. |
| `play_task(task_id, **kwargs) -> None` | Build the task's env (using `play_env` when set), then loop `policy(obs) → env.step` until the viewer closes or `max_steps` is reached. Supports `"zero"` / `"random"` / `"trained"` policies. |
| `AgentKind` | `Literal["zero", "random", "trained"]`. |
| `RslRlOnPolicyRunnerCfg` | Top-level PPO config (extends `RslRlBaseRunnerCfg`); composes `RslRlModelCfg` for actor / critic and `RslRlPpoAlgorithmCfg` for the optimiser. |
| `RslRlVecEnvWrapper` | Adapter from `ManagerBasedRlEnv` to RSL-RL's `VecEnv` protocol. Dynamic subclass of `rsl_rl.env.VecEnv` when RSL-RL is installed. |
| `VecEnvBase` | Minimal stand-alone VecEnv ABC for non-RSL-RL callers. |
| `maybe_profile(...)` | Context manager wrapping `torch.profiler`. No-op when disabled or off rank 0. |
| `profiler_enabled()` | `True` when `GENELAB_PROFILE=1`. |

`genelab.cli` wires `play` / `train` into `play_task` / `train_task` via
`_dispatch_play` / `_dispatch_train` in `src/genelab/cli/__init__.py:384,417`.

## train_task

`train_task` resolves the env cfg off the registered task, applies CLI
overrides, wraps the env, runs PPO, and returns the log directory.

| Argument | Default | Effect |
|---|---|---|
| `task_id: str` | — | Required. Looked up in the `TASKS` registry. |
| `agent_cfg: RslRlOnPolicyRunnerCfg` | — | Required. The CLI pulls this off the task's `cfg.agent`; library callers pass it directly. |
| `num_envs: int \| None` | `None` | Overrides `env_cfg.simulation.num_envs`. |
| `max_iterations: int \| None` | `None` | Overrides `agent_cfg.max_iterations`. |
| `seed: int \| None` | `None` | Sets both `env_cfg.seed` and `agent_cfg.seed`. |
| `log_root: Path \| None` | `Path("logs/rsl_rl")` | Parent under which `<experiment>/<timestamp>` is created. |
| `log_dir: Path \| None` | `None` | Pre-resolved final directory. Takes precedence over `log_root`. The torchrun relaunch path uses this so every rank lands in the same folder. |
| `resume_from: Path \| None` | `None` | Existing checkpoint to load before `runner.learn`. |
| `prof`, `prof_out`, `prof_wait`, `prof_warmup`, `prof_active`, `prof_repeat`, `prof_record_shapes`, `prof_with_stack` | `None` | Forwarded to `maybe_profile`. Each kwarg overrides the matching `GENELAB_PROFILE_*` env var; see "Profiling integration" below. |

Side effects on rank 0:

- Creates `<log_dir>/params/env.json` and `<log_dir>/params/agent.json` with a
  recursive JSON dump of both cfgs.
- RSL-RL writes `<log_dir>/model_<iter>.pt` every `agent_cfg.save_interval`
  iterations.
- TensorBoard event files land under `<log_dir>/` directly.

When profiling is on, `train_task` hooks `RslRlVecEnvWrapper.step` so the
profiler schedule advances once per env step. To express the schedule in PPO
iterations instead, multiply `wait` / `warmup` / `active` by
`agent_cfg.num_steps_per_env`.

## play_task

`play_task` builds the env (preferring `task_cfg.play_env` when set), picks a
policy according to `agent`, then loops until the viewer closes.

| Argument | Default | Effect |
|---|---|---|
| `task_id: str` | — | Required. |
| `checkpoint: Path \| None` | `None` | Required when `agent="trained"`. |
| `num_envs: int \| None` | `None` | Overrides `env_cfg.simulation.num_envs`. |
| `agent: "zero" \| "random" \| "trained" \| None` | `None` | Defaults to `"trained"` if `checkpoint` is given, else `"zero"`. |
| `agent_cfg: RslRlOnPolicyRunnerCfg \| None` | `None` | Required only for `agent="trained"`; falls back to `task.cfg.agent`. |
| `deterministic: bool` | `True` | Forwarded to RSL-RL inference helpers. |
| `max_steps: int \| None` | `None` | Cap the rollout. `None` runs until the viewer closes. |
| `prof*` | `None` | Same semantics as `train_task`. |

The play loop polls `env.viewer_closed` every step, so a closed Genesis
viewer terminates the rollout cleanly without raising. `env.close()` runs in
a `finally` block.

## RslRlOnPolicyRunnerCfg

The top-level dataclass an extension registers as `TaskCfg.agent`. Train
relies on `class_name="OnPolicyRunner"`; an extension config with any other
runner class is rejected at dispatch.

```
RslRlOnPolicyRunnerCfg
├── (inherited from RslRlBaseRunnerCfg)
│   seed=42, num_steps_per_env=24, max_iterations=300
│   obs_groups={"actor": ("policy",), "critic": ("critic",)}
│   save_interval=50, experiment_name="exp1", run_name=""
│   logger="tensorboard" | "wandb"
│   wandb_project="genelab", wandb_tags=()
│   resume=False, load_run=".*", load_checkpoint="model_.*.pt"
│   clip_actions: float | None, upload_model=False
├── class_name="OnPolicyRunner"
├── actor: RslRlModelCfg
├── critic: RslRlModelCfg
└── algorithm: RslRlPpoAlgorithmCfg
```

`RslRlModelCfg` knobs (per actor / critic): `hidden_dims=(128, 128, 128)`,
`activation="elu"`, `obs_normalization=False`, optional `cnn_cfg` /
`distribution_cfg` / `rnn_type` / `rnn_hidden_dim` / `rnn_num_layers`,
`class_name="MLPModel"`. Only `class_name`, `hidden_dims`, `activation`,
`obs_normalization`, and `distribution_cfg` reach the MLP constructor — other
fields are pruned by `_prune_model_cfg`.

`RslRlPpoAlgorithmCfg` knobs: `num_learning_epochs=5`, `num_mini_batches=4`,
`learning_rate=1e-3`, `schedule="adaptive"|"fixed"`, `gamma=0.99`, `lam=0.95`,
`entropy_coef=0.005`, `desired_kl=0.01`, `max_grad_norm=1.0`,
`value_loss_coef=1.0`, `use_clipped_value_loss=True`, `clip_param=0.2`,
`optimizer="adam"|"adamw"|"sgd"|"rmsprop"`, `class_name="PPO"`.

## AgentKind

`AgentKind = Literal["zero", "random", "trained"]`. `play_task` selects the
policy callable from `kind`:

| Kind | Policy | Use case |
|---|---|---|
| `"zero"` | Returns `torch.zeros((num_envs, num_actions))`. | Visual smoke before any training; passive-pose viewer rollouts. |
| `"random"` | Returns `2.0 * rand(...) - 1.0` (uniform `[-1, 1]`). | Sanity-check action bounds and reward shaping. |
| `"trained"` | Loads the checkpoint, instantiates `OnPolicyRunner`, and calls `runner.get_inference_policy()`. | Inspect a trained policy. Requires `checkpoint`. |

## VecEnv wrapper

`RslRlVecEnvWrapper` adapts `ManagerBasedRlEnv` to the interface RSL-RL's
`OnPolicyRunner` expects:

| Attribute / method | Source |
|---|---|
| `num_envs`, `device`, `max_episode_length` | `env.num_envs`, `env.device`, `env.max_episode_length`. |
| `num_actions` | `env.action_manager.total_action_dim`. |
| `num_obs`, `num_privileged_obs` | Last-axis dim of the `"policy"` / `"critic"` observation group, pre-warmed via one `observation_manager.compute()` call at construct. |
| `reset()` | `env.reset()`. Observations are wrapped in a `TensorDict` when `tensordict` is installed; otherwise returned as a plain `dict`. |
| `step(actions)` | Clips actions to `clip_actions` when set, calls `env.step`, packs `(obs, reward, dones, extras)` with `dones = terminated \| truncated` (long) and `extras["time_outs"] = truncated`. |
| `episode_length_buf`, `cfg`, `unwrapped` | Pass-throughs to the underlying env. |

The wrapper dynamically subclasses `rsl_rl.env.VecEnv` at import time via
`_attach_rsl_rl_base()`. When RSL-RL is missing (e.g. unit tests on a
CPU-only host) it falls back to a plain object with the same attributes.

## Distributed training

The CLI surfaces `--gpus N` on `train`. The flag is rewritten by
`_relaunch_under_torchrun` in `cli/__init__.py:457`:

1. Pre-resolves the log directory under `<log_root>/<experiment>/<timestamp>`
   on the parent process so every rank picks up the same path.
2. Strips `--gpus N` from `sys.argv` to avoid an infinite relaunch loop.
3. Appends `--log-dir <pre-resolved>` when the original argv did not already
   carry one.
4. `execvp`s `python -m torch.distributed.run --standalone --nproc_per_node=N
   -m genelab.cli <inner>`.

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 --gpus 4 --num-envs 4096
```

Inside each rank, `genelab.rl.distributed` reads `WORLD_SIZE` / `LOCAL_RANK` /
`RANK` from the environment:

| Helper | Returns |
|---|---|
| `world_size()` | `int(os.environ["WORLD_SIZE"])` or `1`. |
| `local_rank()` | `int(os.environ["LOCAL_RANK"])` or `0`. |
| `global_rank()` | `int(os.environ["RANK"])` or `0`. |
| `is_distributed()` | `world_size() > 1`. |
| `is_main_process()` | `global_rank() == 0`. |
| `pin_cuda_device()` | `"cuda:0"` in distributed runs (the CLI bootstrap pins `CUDA_VISIBLE_DEVICES` per worker, so each rank sees a single device); `None` in single-GPU mode. |

Only the main process writes checkpoints, TensorBoard events, profiler
traces, and the `params/*.json` dump. RSL-RL handles the inter-rank gradient
all-reduce internally.

## Profiling integration

`maybe_profile` wraps `torch.profiler.profile` with a `schedule` and a
TensorBoard trace handler. The CLI surfaces eight flags on both `play` and
`train`:

| CLI flag | `maybe_profile` kwarg | Env var fallback | Default |
|---|---|---|---|
| `--prof` | `enabled` | `GENELAB_PROFILE=1` | off |
| `--prof-out` | `out_dir` | `GENELAB_PROFILE_OUT` | `logs/torch_profile` |
| `--prof-wait` | `wait` | `GENELAB_PROFILE_WAIT` | `10` |
| `--prof-warmup` | `warmup` | `GENELAB_PROFILE_WARMUP` | `5` |
| `--prof-active` | `active` | `GENELAB_PROFILE_ACTIVE` | `10` |
| `--prof-repeat` | `repeat` | `GENELAB_PROFILE_REPEAT` | `2` |
| `--prof-record-shapes` | `record_shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | off |
| `--prof-with-stack` | `with_stack` | `GENELAB_PROFILE_WITH_STACK=1` | off |

The schedule fires one trace per `wait + warmup + active` steps, repeated
`repeat` times. The handler writes `*.pt.trace.json` + `*.tar.gz` under
`out_dir`; `uv run genelab prof open <dir>` launches TensorBoard against it.

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --prof --prof-active 3 --prof-repeat 1 --max-iterations 10
uv run genelab prof open logs/torch_profile
```

## Failure modes worth knowing

!!! warning "Mismatched agent cfg class"
    `_dispatch_train` rejects any `task.cfg.agent` whose type is not
    `RslRlOnPolicyRunnerCfg`. Extensions importing a hand-rolled runner cfg
    that "happens to" have the same fields are not accepted.

!!! warning "agent='trained' requires --checkpoint"
    `play_task` raises `SystemExit("agent='trained' requires a --checkpoint
    path")` when the checkpoint is omitted. Either pass `--checkpoint` or
    pick `--agent zero` / `--agent random` for the visual smoke.

!!! warning "torchrun relaunch loops"
    `_relaunch_under_torchrun` only re-execs when `--gpus N` is set *and*
    `TORCHELASTIC_RUN_ID` is absent. Running `torchrun ... -m genelab.cli
    train --gpus 4` (with `--gpus` left in argv inside the torchrun env)
    skips the relaunch because the env var is now present; this is intended.
    The infinite loop only happens if `--gpus` is forwarded *without*
    torchrun's env vars — typically when wrapping `genelab` inside another
    launcher.

!!! tip "Profile a short run first"
    Profiler traces grow fast. Start with `--prof --prof-active 3
    --prof-repeat 1 --max-iterations 10` to size the trace before running a
    full PPO schedule.

## See also

- [Play and Train](../cli/play-train.md)
- [Profiling](../cli/profiling.md)
- [Managers and MDP terms](managers.md)
