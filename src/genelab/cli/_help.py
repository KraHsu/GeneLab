"""Help-text constants for the CLI commands (split out of ``cli/__init__.py``).

Pure string constants — kept here so the dispatcher module reads as command wiring
rather than prose. Imported by ``cli/__init__.py`` for the ``play`` / ``train``
command ``help=`` arguments.
"""

from typing import Final

# Imported by ``cli/__init__.py`` for the play/train ``help=`` args; declared
# exported so pyright does not flag the cross-module use of these underscore names.
__all__ = ["_PLAY_HELP", "_TRAIN_HELP"]

_RUN_FLAGS_HELP: Final[str] = """\
Shorthand flags rewritten into env overrides:

\b
  -v, --vis        Enable the Genesis viewer (env.simulation.vis=true).
  --headless       Force no viewer (env.simulation.vis=false); mutually
                   exclusive with --vis. Use for `play --agent trained` on a
                   server with no display (its play_env enables the viewer).
  --gpu            Use the GPU backend (env.simulation.gpu=true).
  --steps N        Soft length config (env.simulation.steps=N). Play: headless
                   rollout length; IGNORED with a viewer (--vis runs until you
                   close the window). Train: alias for --max_iterations N.
                   For a hard cap that holds even with a viewer, use --max-steps.
  --dt SECONDS     Override the sim timestep (env.simulation.dt=SECONDS).
  --a.b.c VALUE    Set any dotted cfg path.

Runner flags (used when an RL runner is engaged):

\b
  --num_envs N          Total parallel environments across all ranks.
                        Must divide evenly by --gpus when both are set.
  --num_envs_per_gpu N  Per-rank parallel environments. Mutually exclusive
                        with --num_envs.
  --agent KIND          one of: zero, random, trained (play only).
  --max-steps N         Hard playback cap (play only). Stops after N steps in
                        EITHER mode — wins over --steps and over the viewer gate,
                        so it bounds a run even with a window open. Unset: headless
                        caps at --steps, a viewer runs until the window closes.
  --checkpoint PATH     Resume from a checkpoint.
  --seed N              RNG seed.
  --log_dir PATH        Override the log directory.
  --max_iterations N    Cap training iterations (train only).
  --gpus N              Distributed training across N GPUs (train only).
  --eval_every K        Run a deterministic eval every K iters and save
                        best_model.<ext> on improvement (train only).
  --eval_episodes N     Episodes to roll out per eval (train only, default 10).
  --eval_num_envs N     Envs used during eval (train only, default = train num).
  --eval_seed N         Seed for the eval rollout (train only, default 0).
  --seeds 1,2,3         Fan out into N independent train runs, one per seed
                        (train only). Each child gets --log_dir <parent>/seed_<S>.
  --parallel N          Cap concurrent multi-seed train workers (train only,
                        default 1). Ignored unless --seeds is set.

Profiling flags (forwarded to torch.profiler; rank-0 only):

\b
  --prof                  Enable the profiler (overrides GENELAB_PROFILE).
  --prof-out PATH         TensorBoard trace directory (GENELAB_PROFILE_OUT).
  --prof-wait N           Schedule wait steps (GENELAB_PROFILE_WAIT, default 10).
  --prof-warmup N         Schedule warmup steps (GENELAB_PROFILE_WARMUP, default 5).
  --prof-active N         Schedule active steps (GENELAB_PROFILE_ACTIVE, default 10).
  --prof-repeat N         Schedule cycles (GENELAB_PROFILE_REPEAT, default 2).
  --prof-record-shapes    Record tensor shapes for op input attribution.
  --prof-with-stack       Capture Python stack traces (high overhead).

Use `genelab info TASK` to see the full overridable path list for a task.
Use `genelab prof open [DIR]` to launch TensorBoard against a trace directory.
"""


_PLAY_HELP: Final[str] = "Run a registered task.\n\n" + _RUN_FLAGS_HELP
_TRAIN_HELP: Final[str] = "Train a registered task when a runner exists.\n\n" + _RUN_FLAGS_HELP
