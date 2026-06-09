"""PPO config for the Unitree Go1 velocity tracking task on complex terrain.

Mirrors the flat PPO config with the deltas that keep the terrain curriculum
trainable without de-learning: a per-state action std with an exploration floor
(``HeteroscedasticGaussianDistribution``) and ``entropy_coef = 0``.
"""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go1_velocity_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=3_000,
        save_interval=50,
        experiment_name="go1_velocity_rough",
        logger="tensorboard",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "HeteroscedasticGaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
                "std_range": (0.3, 2.0),
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
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            desired_kl=0.02,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),
    )
