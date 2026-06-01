"""PPO runner configs for the "recall the target" memory task: an LSTM policy + an MLP baseline.

Both train on :func:`inverted_pendulum_memory_env_cfg`. The only difference is the model:
the RNN config sets ``rnn_type="lstm"`` (which ``RslRlModelCfg.__post_init__`` turns into
rsl_rl's ``RNNModel``), giving the policy a hidden state to store the briefly-flashed target.
The MLP baseline is the standard feed-forward config and structurally cannot remember it.

``num_steps_per_env`` is set to span a whole ~100-step episode, so each truncated-BPTT window
covers a full cue→reach cycle and the recurrent gradient can actually learn the recall — the
key trainability detail (a 24-step window left the LSTM unable to associate cue with recall).
"""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from genelab_inverted_pendulum.single.ppo_cfg import inverted_pendulum_ppo_runner_cfg

_ITERATIONS = 400
_STEPS_PER_ENV = 100  # ≈ one episode, so BPTT spans the cue→reach dependency


def _algorithm_cfg() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
    )


def inverted_pendulum_memory_rnn_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """LSTM PPO for the recall task — the recurrent showcase (solves it)."""
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=_STEPS_PER_ENV,
        max_iterations=_ITERATIONS,
        save_interval=50,
        experiment_name="inverted_pendulum_memory_rnn",
        logger="tensorboard",
        clip_actions=100.0,
        actor=RslRlModelCfg(
            hidden_dims=(128,),
            activation="elu",
            obs_normalization=True,
            rnn_type="lstm",
            rnn_hidden_dim=128,
            rnn_num_layers=1,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128,),
            activation="elu",
            obs_normalization=True,
            rnn_type="lstm",
            rnn_hidden_dim=128,
            rnn_num_layers=1,
        ),
        algorithm=_algorithm_cfg(),
    )


def inverted_pendulum_memory_mlp_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Feed-forward (MLP) baseline for the recall task — structurally cannot remember."""
    cfg = inverted_pendulum_ppo_runner_cfg()
    cfg.experiment_name = "inverted_pendulum_memory_mlp"
    cfg.max_iterations = _ITERATIONS
    cfg.num_steps_per_env = _STEPS_PER_ENV
    return cfg
