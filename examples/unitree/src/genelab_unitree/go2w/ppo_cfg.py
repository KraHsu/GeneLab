"""PPO config for the Unitree Go2-W velocity tracking task on flat ground."""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2w_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=6_000,
        save_interval=50,
        experiment_name="go2w_velocity_flat",
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
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
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
            # Mirror data augmentation: the task is mirror-symmetric about the x-z plane,
            # and without this PPO collapses to a one-sided lateral gait (one direction
            # tracks 64/64, the mirror falls 64/64; the surviving side flips between
            # fine-tunes). Mirrored mini-batches make both directions learn in lock-step.
            symmetry_cfg={
                "data_augmentation_func": ("genelab_unitree.go2w.symmetry:mirror_go2w_obs_actions"),
                "use_data_augmentation": True,
            },
        ),
    )
