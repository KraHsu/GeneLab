"""PPO runner config for the Franka pick-and-place task."""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def franka_pick_and_place_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=500,
        save_interval=50,
        experiment_name="franka_pick_and_place",
        logger="tensorboard",
        clip_actions=1.0,
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                # Hard-cap the policy std. rsl_rl clamps std to this range on
                # every update; the default (1e-6, 1e6) leaves it unbounded.
                # With clip_actions=1.0, samples beyond [-1, 1] are clipped
                # before reaching the env, so once std grows large the surrogate
                # loss is insensitive to it and the entropy bonus inflates std
                # without limit (observed: 1.0 -> 9.2 and rising). Capping at
                # the init value keeps exploration strong without runaway.
                "std_range": (0.1, 1.0),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            # Lowered from 0.01: with the std cap in place the policy can now
            # actually reduce std to exploit instead of being pinned at the cap.
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),
    )
