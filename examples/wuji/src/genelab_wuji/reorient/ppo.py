"""RSL-RL PPO config for the WUJI-hand reorientation task (mirrors the mjlab reference)."""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def wuji_hand_reorient_ppo_runner_cfg(max_iterations: int = 1500) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=40,
        max_iterations=max_iterations,
        save_interval=50,
        experiment_name="wuji_reorient",
        logger="tensorboard",
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_range": (0.2, 1.0e6),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=False,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=4,
            num_mini_batches=32,
            learning_rate=1.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
        ),
    )
