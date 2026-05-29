"""PPO config for the Unitree G1 velocity tracking task on rough terrain.

Identical to the flat velocity PPO config (``ppo_cfg.unitree_g1_ppo_runner_cfg``)
except for the iteration budget and the experiment name. The network, the
algorithm hyperparameters, the RNG seed, and the logger all match the flat task
so the two runs are comparable.
"""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_g1_velocity_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        # Why: rough terrain converges slower than flat; Isaac Lab budgets 50k
        # iterations for the equivalent Velocity-Rough-G1 task.
        max_iterations=50_000,
        save_interval=50,
        experiment_name="g1_velocity_rough",
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
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),
    )
