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
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from genelab.bridges.base import Bridge
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

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


@runtime_checkable
class Backend(Protocol):
    """One RL library. Implementations register themselves via ``register_backend``."""

    name: str
    cfg_type: type

    def train(self, ctx: TrainContext) -> Path:
        """Train the task in ``ctx`` and return the log directory."""
        ...

    def play(self, ctx: PlayContext) -> None:
        """Replay a policy for the task in ``ctx``."""
        ...
