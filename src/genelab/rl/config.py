"""RSL-RL configuration dataclasses (mirror of ``rl.config`` with tensorboard default)."""

from dataclasses import dataclass, field
from typing import Any, Literal


class BackendConfig:
    """Marker base for the agent/runner config an RL backend dispatches on.

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

    def __post_init__(self) -> None:
        # ``rnn_type`` is the single knob for a recurrent policy: setting it selects
        # rsl_rl's ``RNNModel`` automatically so callers don't also have to flip
        # ``class_name`` (which stays an internal rsl_rl detail). An explicit non-default
        # ``class_name`` (e.g. ``"CNNModel"``) is left untouched.
        if self.rnn_type is not None:
            if self.rnn_type not in ("lstm", "gru"):
                raise ValueError(
                    f"rnn_type must be 'lstm' or 'gru' (or None for an MLP policy); "
                    f"got {self.rnn_type!r}"
                )
            if self.class_name == "MLPModel":
                self.class_name = "RNNModel"
        elif self.class_name == "RNNModel":
            # ``RNNModel`` needs a concrete rnn_type; without one rsl_rl would crash deep
            # in ``RNN.__init__`` (``None.lower()``). Fail early with a clear message.
            raise ValueError("class_name='RNNModel' requires rnn_type='lstm' or 'gru'")


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
    # rsl_rl mirror-symmetry extension (Mittal et al., ICRA 2024): dict with
    # ``data_augmentation_func`` ("module:func" or callable), ``use_data_augmentation``,
    # ``use_mirror_loss``, ``mirror_loss_coeff``. Tasks with a mirror-symmetric morphology
    # (e.g. Go2-W lateral gait) set it so both mirror directions learn in lock-step instead
    # of collapsing to a one-sided gait. ``None`` (default) disables the extension.
    symmetry_cfg: dict[str, Any] | None = None
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
