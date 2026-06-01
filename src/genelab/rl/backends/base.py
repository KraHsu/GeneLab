"""Backend abstraction shared by every RL library GeneLab can train through.

A *backend* owns one RL library. It is selected by the type of the task's agent
config (see :func:`genelab.rl.backends.select_backend`): ``RslRlOnPolicyRunnerCfg``
routes to the RSL-RL backend, ``SkrlAgentCfg`` to the skrl backend, and so on.

``runner.train_task`` / ``runner.play_task`` build a :class:`TrainContext` /
:class:`PlayContext` (env, bridges, profiler knobs already resolved) and hand it to
the chosen backend, so library-specific code lives entirely inside the backend.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch.nn as nn

    from genelab.bridges.base import Bridge
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.rl.config import BackendConfig

AgentKind = Literal["zero", "random", "trained"]


@dataclass
class ProfileArgs:
    """The eight ``GENELAB_PROFILE_*`` overrides, bundled so contexts stay readable.

    Every field mirrors a keyword argument of :func:`genelab.rl.profiler.maybe_profile`;
    ``None`` means "fall back to the env var / built-in default".
    """

    prof: bool | None = None
    prof_out: Path | None = None
    prof_wait: int | None = None
    prof_warmup: int | None = None
    prof_active: int | None = None
    prof_repeat: int | None = None
    prof_record_shapes: bool | None = None
    prof_with_stack: bool | None = None

    def as_maybe_profile_kwargs(self) -> dict[str, Any]:
        """Return the kwargs ``maybe_profile`` expects (its names drop the ``prof_`` prefix)."""
        return {
            "enabled": self.prof,
            "out_dir": self.prof_out,
            "wait": self.prof_wait,
            "warmup": self.prof_warmup,
            "active": self.prof_active,
            "repeat": self.prof_repeat,
            "record_shapes": self.prof_record_shapes,
            "with_stack": self.prof_with_stack,
        }


@dataclass
class TrainContext:
    """Everything a backend needs to train a task.

    The dispatcher has already applied ``num_envs`` / ``seed`` to ``env_cfg`` and
    built ``env``; ``max_iterations`` / ``seed`` are passed through verbatim so each
    backend can map them onto its own config (RSL-RL iterations vs skrl timesteps).
    """

    task_id: str
    env: "ManagerBasedRlEnv"
    env_cfg: Any
    agent_cfg: Any
    max_iterations: int | None = None
    seed: int | None = None
    log_dir: Path | None = None
    log_root: Path | None = None
    resume_from: Path | None = None
    profile: ProfileArgs = field(default_factory=ProfileArgs)
    # Periodic in-training eval (M1.2). Driven by runner.train_task, not by the backend.
    eval_callback: Any = None  # genelab.rl.eval_callback.EvalCallbackCfg | None


@dataclass
class PlayContext:
    """Everything a backend needs to replay a policy.

    ``bridges`` have already been instantiated and had ``on_build`` called; the
    backend runs the rollout loop and is responsible for ``on_close`` teardown
    (``runner._run_play_loop`` + ``runner._close_bridges`` cover this).
    """

    task_id: str
    env: "ManagerBasedRlEnv"
    env_cfg: Any
    agent_cfg: Any | None = None
    checkpoint: Path | None = None
    kind: AgentKind = "zero"
    deterministic: bool = True
    max_steps: int | None = None
    bridges: list["Bridge"] = field(default_factory=list)
    profile: ProfileArgs = field(default_factory=ProfileArgs)


@dataclass
class InferenceSetup:
    """Bundle of pieces ``run_play_loop`` / ``run_evaluation`` / ``export_policy`` consume.

    Built by ``Backend.make_inference_setup`` so eval / play / export share one
    code path that already knows how to (a) wrap the env, (b) load a checkpoint, and
    (c) hand back the actor as a bare ``nn.Module`` for export.

    Fields:

    * ``wrapper`` — the backend-specific env wrapper (``RslRlVecEnvWrapper`` /
      ``GenelabSkrlWrapper`` / ``GenelabSb3VecEnv``). Holds ``.device``, ``.num_envs``,
      ``.num_actions``, etc.
    * ``adapter`` — a ``run_play_loop`` / ``run_evaluation`` compatible env-like object
      whose ``reset`` returns ``(obs, info)`` and ``step`` returns ``(obs, reward,
      terminated, truncated, info)``. The info dict carries ``extras["is_success"]``
      when the task exposes it.
    * ``policy`` — ``policy(obs) -> action_tensor`` callable for the rollout loop.
    * ``actor_module`` — bare ``nn.Module`` taking a single flat policy-obs tensor
      and emitting an action tensor. ``None`` when the backend cannot extract it
      (e.g. zero / random play paths). Required for ``genelab export``. For a recurrent
      policy this is the raw model exposing ``as_jit()`` / ``as_onnx()`` (the exporter
      handles the hidden-state wiring); see ``is_recurrent``.
    * ``actor_input_dim`` — flat dim of the policy obs tensor the actor expects.
      Used to build dummy inputs for tracing / ONNX export.
    * ``is_recurrent`` — ``True`` when the policy carries hidden state (RNN/LSTM/GRU).
      The exporter then uses the recurrent export path, and the play / eval loops call
      ``reset_hidden`` on episode boundaries.
    * ``reset_hidden`` — ``reset_hidden(dones)`` zeros the policy's hidden state for the
      environments whose episode just ended. ``None`` for stateless policies (the loops
      then skip it). Always paired with ``is_recurrent``.
    """

    wrapper: Any
    adapter: Any
    policy: Callable[[Any], Any]
    actor_module: "nn.Module | None" = None
    actor_input_dim: int | None = None
    is_recurrent: bool = False
    reset_hidden: Callable[[Any], None] | None = None


@runtime_checkable
class Backend(Protocol):
    """One RL library. Implementations register themselves via ``register_backend``."""

    name: str

    @property
    def cfg_type(self) -> "type[BackendConfig]":
        """The ``BackendConfig`` subclass this backend is dispatched on.

        Read-only (a property) so a concrete ``cfg_type = SkrlAgentCfg`` class attribute
        satisfies it covariantly — a mutable attribute would be invariant on the type arg.
        """
        ...

    def train(self, ctx: TrainContext) -> Path:
        """Train the task in ``ctx`` and return the log directory."""
        ...

    def play(self, ctx: PlayContext) -> None:
        """Replay a policy for the task in ``ctx``."""
        ...

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        """Load ``ctx.checkpoint`` and return a play / eval / export-ready bundle.

        Always assumes ``ctx.kind == "trained"`` — zero / random play do not need an
        ``InferenceSetup`` because they have no checkpoint to load.
        """
        ...
