"""PPO config for the Unitree Go1 velocity tracking task on flat ground."""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        # 1500 was too short for the quadruped to master both directions — forward emerged
        # but backward locomotion (learned last) stayed weak. 3000 gives it time to develop
        # symmetric velocity tracking.
        max_iterations=3_000,
        save_interval=50,
        experiment_name="go1_velocity_flat",
        logger="tensorboard",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            # A plain global Gaussian std (flat ground has no terrain curriculum to
            # de-learn against, unlike the rough task's heteroscedastic head). rsl_rl
            # requires an explicit distribution — without it self.distribution stays None
            # and the first act() raises on log_prob.
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
