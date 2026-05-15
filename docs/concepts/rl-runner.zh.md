# RL runner

`genelab.rl` 是把已注册 task 跑成 PPO 训练 / 回放会话的薄 RSL-RL 集成层。CLI 子命令
`play` 与 `train` 直接派发到下面两个入口；库调用者也可直接用。

## 模块表面

| 名字 | 用途 |
|---|---|
| `train_task(task_id, agent_cfg, **kwargs) -> Path` | 构造 task 的 env、用 `RslRlVecEnvWrapper` 包装、驱动 RSL-RL 的 `OnPolicyRunner.learn`。返回解析后的 log 目录。 |
| `play_task(task_id, **kwargs) -> None` | 构造 task 的 env（设置了 `play_env` 时优先用它），按 `agent` 选 policy，循环 `policy(obs) → env.step` 直到 viewer 关闭或达到 `max_steps`。支持 `"zero"` / `"random"` / `"trained"` 三种 policy。 |
| `AgentKind` | `Literal["zero", "random", "trained"]`。 |
| `RslRlOnPolicyRunnerCfg` | 顶层 PPO 配置（继承 `RslRlBaseRunnerCfg`）；组合 actor / critic 的 `RslRlModelCfg` 与优化器的 `RslRlPpoAlgorithmCfg`。 |
| `RslRlVecEnvWrapper` | 把 `ManagerBasedRlEnv` 适配到 RSL-RL 的 `VecEnv` 协议。装了 RSL-RL 时动态成为 `rsl_rl.env.VecEnv` 的子类。 |
| `VecEnvBase` | 给非 RSL-RL 调用方的最小 VecEnv ABC。 |
| `maybe_profile(...)` | `torch.profiler` 的上下文管理器。关闭或非 rank 0 时 no-op。 |
| `profiler_enabled()` | `GENELAB_PROFILE=1` 时返回 `True`。 |

`genelab.cli` 把 `play` / `train` 通过 `src/genelab/cli/__init__.py:384,417` 中的
`_dispatch_play` / `_dispatch_train` 接到 `play_task` / `train_task`。

## train_task

`train_task` 从已注册 task 上解析 env cfg、应用 CLI override、包装 env、跑 PPO 并
返回 log 目录。

| 参数 | 默认 | 行为 |
|---|---|---|
| `task_id: str` | — | 必填。在 `TASKS` 注册表里查找。 |
| `agent_cfg: RslRlOnPolicyRunnerCfg` | — | 必填。CLI 从 task 的 `cfg.agent` 上拿；库调用者直接传。 |
| `num_envs: int \| None` | `None` | 覆盖 `env_cfg.simulation.num_envs`。 |
| `max_iterations: int \| None` | `None` | 覆盖 `agent_cfg.max_iterations`。 |
| `seed: int \| None` | `None` | 同时设置 `env_cfg.seed` 与 `agent_cfg.seed`。 |
| `log_root: Path \| None` | `Path("logs/rsl_rl")` | 在该目录下创建 `<experiment>/<timestamp>`。 |
| `log_dir: Path \| None` | `None` | 预解析的最终目录，优先级高于 `log_root`。torchrun 重启路径用这个让每个 rank 落在同一文件夹。 |
| `resume_from: Path \| None` | `None` | `runner.learn` 之前要加载的已有 checkpoint。 |
| `prof`、`prof_out`、`prof_wait`、`prof_warmup`、`prof_active`、`prof_repeat`、`prof_record_shapes`、`prof_with_stack` | `None` | 转发给 `maybe_profile`。每个 kwarg 覆盖对应的 `GENELAB_PROFILE_*` env var；参见下面 "Profiler 集成" 节。 |

Rank 0 的副作用：

- 在 `<log_dir>/params/env.json` 与 `<log_dir>/params/agent.json` 中递归 JSON dump
  两份 cfg。
- RSL-RL 按 `agent_cfg.save_interval` 写 `<log_dir>/model_<iter>.pt`。
- TensorBoard event 直接落到 `<log_dir>/` 下。

打开 profiler 时，`train_task` 会 hook `RslRlVecEnvWrapper.step`，使 profiler
schedule 每 env step 推进一次。若想以 PPO iteration 表达 schedule，把
`wait` / `warmup` / `active` 乘以 `agent_cfg.num_steps_per_env`。

## play_task

`play_task` 构造 env（设置 `task_cfg.play_env` 时优先），按 `agent` 选 policy，循环
直到 viewer 关闭。

| 参数 | 默认 | 行为 |
|---|---|---|
| `task_id: str` | — | 必填。 |
| `checkpoint: Path \| None` | `None` | `agent="trained"` 时必填。 |
| `num_envs: int \| None` | `None` | 覆盖 `env_cfg.simulation.num_envs`。 |
| `agent: "zero" \| "random" \| "trained" \| None` | `None` | 给了 `checkpoint` 默认 `"trained"`，否则 `"zero"`。 |
| `agent_cfg: RslRlOnPolicyRunnerCfg \| None` | `None` | 只有 `agent="trained"` 才需要；缺省 fallback 到 `task.cfg.agent`。 |
| `deterministic: bool` | `True` | 传给 RSL-RL 推理辅助函数。 |
| `max_steps: int \| None` | `None` | rollout 上限。`None` 表示跑到 viewer 关闭。 |
| `prof*` | `None` | 语义同 `train_task`。 |

Play 循环每步检测 `env.viewer_closed`，关 Genesis viewer 时干净退出。`env.close()`
放在 `finally` 块里。

## RslRlOnPolicyRunnerCfg

扩展通过 `TaskCfg.agent` 注册的顶层 dataclass。`train` 依赖
`class_name="OnPolicyRunner"`；扩展 cfg 用了其他 runner 类会在 dispatch 时被拒。

```
RslRlOnPolicyRunnerCfg
├──（继承自 RslRlBaseRunnerCfg）
│   seed=42、num_steps_per_env=24、max_iterations=300
│   obs_groups={"actor": ("policy",), "critic": ("critic",)}
│   save_interval=50、experiment_name="exp1"、run_name=""
│   logger="tensorboard" | "wandb"
│   wandb_project="genelab"、wandb_tags=()
│   resume=False、load_run=".*"、load_checkpoint="model_.*.pt"
│   clip_actions: float | None、upload_model=False
├── class_name="OnPolicyRunner"
├── actor: RslRlModelCfg
├── critic: RslRlModelCfg
└── algorithm: RslRlPpoAlgorithmCfg
```

`RslRlModelCfg` 旋钮（actor / critic 各一份）：`hidden_dims=(128, 128, 128)`、
`activation="elu"`、`obs_normalization=False`，可选 `cnn_cfg` / `distribution_cfg` /
`rnn_type` / `rnn_hidden_dim` / `rnn_num_layers`、`class_name="MLPModel"`。
只有 `class_name`、`hidden_dims`、`activation`、`obs_normalization`、
`distribution_cfg` 会进入 MLP 构造函数；其余字段被 `_prune_model_cfg` 过滤掉。

`RslRlPpoAlgorithmCfg` 旋钮：`num_learning_epochs=5`、`num_mini_batches=4`、
`learning_rate=1e-3`、`schedule="adaptive"|"fixed"`、`gamma=0.99`、`lam=0.95`、
`entropy_coef=0.005`、`desired_kl=0.01`、`max_grad_norm=1.0`、
`value_loss_coef=1.0`、`use_clipped_value_loss=True`、`clip_param=0.2`、
`optimizer="adam"|"adamw"|"sgd"|"rmsprop"`、`class_name="PPO"`。

## AgentKind

`AgentKind = Literal["zero", "random", "trained"]`。`play_task` 按 `kind` 挑 policy
可调用：

| Kind | Policy | 适用场景 |
|---|---|---|
| `"zero"` | 返回 `torch.zeros((num_envs, num_actions))`。 | 训练前的视觉 smoke；被动姿态 viewer rollout。 |
| `"random"` | 返回 `2.0 * rand(...) - 1.0`（uniform `[-1, 1]`）。 | sanity-check action 边界与 reward shaping。 |
| `"trained"` | 加载 checkpoint，实例化 `OnPolicyRunner`，调 `runner.get_inference_policy()`。 | 看训好的 policy。需要 `checkpoint`。 |

## VecEnv wrapper

`RslRlVecEnvWrapper` 把 `ManagerBasedRlEnv` 适配到 RSL-RL 的 `OnPolicyRunner`
期望的接口：

| 属性 / 方法 | 来源 |
|---|---|
| `num_envs`、`device`、`max_episode_length` | `env.num_envs`、`env.device`、`env.max_episode_length`。 |
| `num_actions` | `env.action_manager.total_action_dim`。 |
| `num_obs`、`num_privileged_obs` | `"policy"` / `"critic"` 观测组的末轴维数，构造时先跑一次 `observation_manager.compute()` 取出。 |
| `reset()` | `env.reset()`。装了 `tensordict` 时把 obs 包成 `TensorDict`；否则原样返回 plain `dict`。 |
| `step(actions)` | 设了 `clip_actions` 就先 clip，调 `env.step`，打包 `(obs, reward, dones, extras)`，其中 `dones = terminated \| truncated`（long）、`extras["time_outs"] = truncated`。 |
| `episode_length_buf`、`cfg`、`unwrapped` | 透传到底层 env。 |

Wrapper 在 import 时通过 `_attach_rsl_rl_base()` 动态成为 `rsl_rl.env.VecEnv`
的子类。没装 RSL-RL（例如 CPU-only 主机上的单元测试）时回退到拥有同样属性的 plain
对象。

## 分布式训练

CLI 的 `train` 上有 `--gpus N` 旗标。`cli/__init__.py:457` 的
`_relaunch_under_torchrun` 在父进程里改写它：

1. 在父进程预解析 log 目录为 `<log_root>/<experiment>/<timestamp>`，让每个 rank
   拿到同一路径。
2. 从 `sys.argv` 里剥掉 `--gpus N` 避免重启循环。
3. 原始 argv 里没带 `--log-dir` 时追加 `--log-dir <预解析路径>`。
4. `execvp`：`python -m torch.distributed.run --standalone --nproc_per_node=N
   -m genelab.cli <inner>`。

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 --gpus 4 --num-envs 4096
```

每个 rank 内部，`genelab.rl.distributed` 从环境读 `WORLD_SIZE` / `LOCAL_RANK` /
`RANK`：

| Helper | 返回 |
|---|---|
| `world_size()` | `int(os.environ["WORLD_SIZE"])` 或 `1`。 |
| `local_rank()` | `int(os.environ["LOCAL_RANK"])` 或 `0`。 |
| `global_rank()` | `int(os.environ["RANK"])` 或 `0`。 |
| `is_distributed()` | `world_size() > 1`。 |
| `is_main_process()` | `global_rank() == 0`。 |
| `pin_cuda_device()` | 分布式 run 中返回 `"cuda:0"`（CLI bootstrap 已按 rank 钉 `CUDA_VISIBLE_DEVICES`，每个 rank 只看见一张卡）；单 GPU 模式返回 `None`。 |

只有 main process 写 checkpoint、TensorBoard event、profiler trace 与
`params/*.json` 转储。RSL-RL 内部跨 rank all-reduce 梯度。

## Profiler 集成

`maybe_profile` 用一个 `schedule` 与 TensorBoard trace handler 包住
`torch.profiler.profile`。CLI 在 `play` 与 `train` 上各暴露 8 个 flag：

| CLI flag | `maybe_profile` kwarg | Env var fallback | 默认 |
|---|---|---|---|
| `--prof` | `enabled` | `GENELAB_PROFILE=1` | off |
| `--prof-out` | `out_dir` | `GENELAB_PROFILE_OUT` | `logs/torch_profile` |
| `--prof-wait` | `wait` | `GENELAB_PROFILE_WAIT` | `10` |
| `--prof-warmup` | `warmup` | `GENELAB_PROFILE_WARMUP` | `5` |
| `--prof-active` | `active` | `GENELAB_PROFILE_ACTIVE` | `10` |
| `--prof-repeat` | `repeat` | `GENELAB_PROFILE_REPEAT` | `2` |
| `--prof-record-shapes` | `record_shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | off |
| `--prof-with-stack` | `with_stack` | `GENELAB_PROFILE_WITH_STACK=1` | off |

Schedule 每 `wait + warmup + active` 步出一份 trace，重复 `repeat` 次。Handler 在
`out_dir` 下写 `*.pt.trace.json` + `*.tar.gz`；`uv run genelab prof open <dir>`
对其起 TensorBoard。

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --prof --prof-active 3 --prof-repeat 1 --max-iterations 10
uv run genelab prof open logs/torch_profile
```

## 已知失效模式

!!! warning "agent cfg 类型不匹配"
    `_dispatch_train` 拒绝任何 `task.cfg.agent` 类型不是 `RslRlOnPolicyRunnerCfg`
    的 task。"字段刚好一样"的自造 runner cfg 不被接受。

!!! warning "agent='trained' 需要 --checkpoint"
    缺 checkpoint 时 `play_task` 抛 `SystemExit("agent='trained' requires a
    --checkpoint path")`。要么传 `--checkpoint`，要么挑 `--agent zero` /
    `--agent random` 做视觉 smoke。

!!! warning "torchrun 重启回环"
    `_relaunch_under_torchrun` 仅在 `--gpus N` 出现 *且* `TORCHELASTIC_RUN_ID`
    缺失时重新 exec。手工跑 `torchrun ... -m genelab.cli train --gpus 4`（在
    torchrun 内部 argv 里仍保留 `--gpus`）会跳过重启 —— 因为 env var 已存在；这是
    预期行为。无限循环只在 `--gpus` 被转发但 *没有* torchrun 环境变量时出现，
    通常是 `genelab` 被其他 launcher 套进去导致的。

!!! tip "先 profile 一段短 run"
    Profiler trace 增长很快。从 `--prof --prof-active 3 --prof-repeat 1
    --max-iterations 10` 起步，先看 trace 体量再跑完整 PPO schedule。

## See also

- [Play and Train](../cli/play-train.md)
- [Profiling](../cli/profiling.md)
- [Managers and MDP terms](managers.md)
