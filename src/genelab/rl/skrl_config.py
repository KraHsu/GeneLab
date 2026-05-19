"""skrl agent configuration dataclasses.

A single :class:`SkrlAgentCfg` covers every supported algorithm (PPO, A2C, SAC,
TD3, DDPG): only the fields relevant to the selected ``algorithm`` are consumed
when the backend builds the skrl agent. ``extra_cfg`` is an escape hatch merged
last over the algorithm's ``<ALGO>_DEFAULT_CONFIG``.

These dataclasses deliberately import nothing from skrl, so ``genelab.rl`` stays
importable without skrl installed.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

SkrlAlgorithm = Literal["PPO", "A2C", "SAC", "TD3", "DDPG"]

#: Algorithms that learn from a rollout buffer (use ``rollouts`` as memory size).
ON_POLICY_ALGORITHMS: frozenset[str] = frozenset({"PPO", "A2C"})
#: Algorithms that learn from a replay buffer (use ``replay_buffer_size``).
OFF_POLICY_ALGORITHMS: frozenset[str] = frozenset({"SAC", "TD3", "DDPG"})


@dataclass
class SkrlModelCfg:
    """Network architecture shared by every policy / value / critic model."""

    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "elu"
    # Gaussian-policy only (PPO/A2C/SAC); ignored for deterministic models.
    initial_log_std: float = 0.0
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    clip_actions: bool = False


@dataclass
class SkrlExperimentCfg:
    """Logging / checkpoint settings — maps onto skrl's agent ``cfg["experiment"]``."""

    experiment_name: str = "exp1"
    run_name: str = ""
    # "auto" lets skrl pick a sane cadence; an int forces a fixed step interval.
    write_interval: int | Literal["auto"] = "auto"
    checkpoint_interval: int | Literal["auto"] = "auto"
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "genelab"
    wandb_tags: tuple[str, ...] = ()


@dataclass
class SkrlAgentCfg:
    """skrl runner config. Selected by the backend dispatcher via ``type(agent_cfg)``."""

    algorithm: SkrlAlgorithm = "PPO"
    seed: int = 42
    # skrl trains in environment timesteps, not learning iterations.
    timesteps: int = 100_000
    rollouts: int = 24  # on-policy memory size (steps per update)
    replay_buffer_size: int = 1_000_000  # off-policy replay buffer capacity
    learning_rate: float = 3.0e-4
    discount_factor: float = 0.99
    # On-policy (PPO / A2C).
    learning_epochs: int = 5
    mini_batches: int = 4
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    grad_norm_clip: float = 1.0
    # Off-policy (SAC / TD3 / DDPG).
    batch_size: int = 256
    polyak: float = 0.005
    learning_starts: int = 1000
    # Escape hatch: merged last over the algorithm's skrl DEFAULT_CONFIG.
    extra_cfg: dict[str, Any] = field(default_factory=dict)
    models: SkrlModelCfg = field(default_factory=SkrlModelCfg)
    experiment: SkrlExperimentCfg = field(default_factory=SkrlExperimentCfg)
    # Which observation-manager groups feed the agent and the (optional) critic state.
    obs_group: str = "policy"
    state_group: str | None = "critic"

    @property
    def is_on_policy(self) -> bool:
        return self.algorithm in ON_POLICY_ALGORITHMS
