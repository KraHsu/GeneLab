"""RSL-RL configuration dataclasses (mirror of ``rl.config`` with tensorboard default)."""

from dataclasses import dataclass, field
from typing import Any, Literal


class BackendConfig:
    """Marker base for the agent/runner config an RL backend dispatches on (ADR-0016 step 2).

    Every config ``select_backend`` keys on subclasses this, so the backend registry
    is typed (``dict[type[BackendConfig], Backend]``) rather than keyed on a bare
    ``type``. It lives here — not in ``rl/backends/base.py`` — because rsl_rl's dispatch
    config (``RslRlOnPolicyRunnerCfg``) is defined in this module and ``rl.backends``
    already imports it; keeping the base here keeps every edge pointing down to
    ``rl.config`` and avoids a ``config ↔ backends`` import cycle.
    """


@dataclass
class RslRlModelCfg:
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    obs_normalization: bool = False
    cnn_cfg: dict[str, Any] | None = None
    distribution_cfg: dict[str, Any] | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    class_name: str = "MLPModel"


@dataclass
class RslRlPpoAlgorithmCfg:
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    share_cnn_encoders: bool = False
    class_name: str = "PPO"


@dataclass
class RslRlBaseRunnerCfg(BackendConfig):
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 300
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("policy",), "critic": ("critic",)}
    )
    save_interval: int = 50
    experiment_name: str = "exp1"
    run_name: str = ""
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    wandb_project: str = "genelab"
    wandb_tags: tuple[str, ...] = ()
    clip_actions: float | None = None
    upload_model: bool = False


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
