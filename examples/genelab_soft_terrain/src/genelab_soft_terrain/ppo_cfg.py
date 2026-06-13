"""PPO configs for the soft-terrain velocity-tracking tasks."""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def g1_mattress_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """The Go1 soft/sand recipe with the G1-specific exploration fixes.

    The Go1 ``entropy_coef=0.005`` collapses on the 29-DoF humanoid: by iteration
    ~1250 the action std had fallen 1.0 -> 0.14 and the policy was stuck in a
    stand-then-tip local optimum (feet_air_time ~ 0 — never stepping). The proven
    G1 flat-ground recipe (``genelab_unitree.g1.ppo_cfg``) uses 0.01 for exactly
    this reason, and budgets tens of thousands of iterations for a humanoid gait.
    """
    cfg = go1_soft_velocity_ppo_runner_cfg()
    cfg.experiment_name = "g1_mattress_velocity"
    cfg.max_iterations = 10_000
    cfg.algorithm.entropy_coef = 0.01
    return cfg


def go1_soft_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=3_000,
        save_interval=50,
        experiment_name="go1_soft_velocity",
        logger="tensorboard",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
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
        ),
    )
