"""Stable-Baselines3 (SB3) agent configuration dataclasses.

A single :class:`Sb3AgentCfg` covers every supported algorithm (PPO, A2C, SAC,
TD3, DDPG): only the fields relevant to the selected ``algorithm`` are consumed
when the backend builds the SB3 model. ``extra_kwargs`` is an escape hatch merged
last over the algorithm constructor's keyword arguments.

:class:`Sb3HerCfg` opts a task into Hindsight Experience Replay: with ``enabled``
the wrapper exposes a goal-conditioned ``Dict`` observation and the off-policy
algorithm trains through SB3's ``HerReplayBuffer``.

These dataclasses deliberately import nothing from ``stable_baselines3`` (or
torch), so ``genelab.rl`` stays importable without SB3 installed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from genelab.rl._algorithm_taxonomy import (
    OFF_POLICY_ALGORITHMS as OFF_POLICY_ALGORITHMS,  # explicit re-export
    ON_POLICY_ALGORITHMS,
)
from genelab.rl.config import BackendConfig

Sb3Algorithm = Literal["PPO", "A2C", "SAC", "TD3", "DDPG"]


@dataclass
class Sb3PolicyCfg:
    """Network architecture — maps onto SB3's ``policy_kwargs``.

    ``net_arch`` sizes the shared MLP trunk; ``activation`` is resolved to a
    ``torch.nn`` module by the backend (kept as a string so this module stays
    torch-free).
    """

    net_arch: tuple[int, ...] = (256, 256)
    activation: str = "elu"


@dataclass
class Sb3ExperimentCfg:
    """Logging / checkpoint settings for an SB3 run."""

    experiment_name: str = "exp1"
    run_name: str = ""
    # Env-steps between ``CheckpointCallback`` dumps (0 disables periodic dumps;
    # the final model is always saved at the end of training).
    checkpoint_interval: int = 0
    logger: Literal["tensorboard", "stdout"] = "tensorboard"


@dataclass
class Sb3HerCfg:
    """Hindsight Experience Replay settings.

    When ``enabled``, the SB3 wrapper exposes a goal-conditioned ``Dict``
    observation built from three observation-manager groups, and the off-policy
    algorithm trains through ``HerReplayBuffer``. ``compute_reward`` is the task's
    goal-relabelling reward — ``compute_reward(achieved_goal, desired_goal, info)``
    over numpy batches — required by HER and unavailable from the env's reward
    manager (which scores full state, not goal pairs).
    """

    enabled: bool = False
    obs_group: str = "policy"
    achieved_goal_group: str = "achieved_goal"
    desired_goal_group: str = "desired_goal"
    n_sampled_goal: int = 4
    goal_selection_strategy: Literal["future", "final", "episode"] = "future"
    compute_reward: Callable[..., Any] | None = None


@dataclass
class Sb3AgentCfg(BackendConfig):
    """SB3 runner config. Selected by the backend dispatcher via ``type(agent_cfg)``."""

    algorithm: Sb3Algorithm = "PPO"
    seed: int = 42
    # SB3 trains in environment timesteps, not learning iterations.
    total_timesteps: int = 100_000
    learning_rate: float = 3.0e-4
    discount_factor: float = 0.99
    batch_size: int = 256
    # On-policy (PPO / A2C).
    n_steps: int = 24  # rollout-buffer size, per env
    n_epochs: int = 5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    # Off-policy (SAC / TD3 / DDPG).
    buffer_size: int = 1_000_000
    tau: float = 0.005
    learning_starts: int = 1000
    train_freq: int = 1
    # Escape hatch: merged last over the algorithm constructor's keyword arguments.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    policy: Sb3PolicyCfg = field(default_factory=Sb3PolicyCfg)
    experiment: Sb3ExperimentCfg = field(default_factory=Sb3ExperimentCfg)
    her: Sb3HerCfg = field(default_factory=Sb3HerCfg)
    # Observation-manager group feeding the agent (flat, non-HER mode only).
    obs_group: str = "policy"
    # Path to a ``.npz`` of offline demonstrations to pre-load into the replay
    # buffer before ``model.learn``. Off-policy algorithms only; HER-aware. The
    # file shape matches ``genelab_franka.collect_demos``. The
    # backend also honors the ``GENELAB_SB3_DEMO_PATH`` env var when this is
    # ``None`` so the CLI can opt in without touching cfg.
    demo_path: str | None = None

    @property
    def is_on_policy(self) -> bool:
        return self.algorithm in ON_POLICY_ALGORITHMS
