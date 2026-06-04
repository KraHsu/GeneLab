"""PPO config for the Unitree G1 velocity tracking task on rough terrain.

Mirrors the flat velocity PPO config (``ppo_cfg.unitree_g1_ppo_runner_cfg``) with the
deltas that make the terrain curriculum trainable without de-learning: a per-state
action std with an exploration floor (see ``distribution_cfg`` below) and
``entropy_coef = 0``. The network, the RNG seed and the logger match the flat task.
"""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_g1_velocity_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        # With the de-learning fixed (per-state std floor + entropy_coef 0) the policy
        # trains stably to convergence across seeds — action std holds at its 0.3 floor,
        # reward plateaus, and the terrain curriculum climbs without the run de-learning,
        # so no early-stop crutch is needed.
        max_iterations=6_000,
        save_interval=50,
        experiment_name="g1_velocity_rough",
        logger="tensorboard",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            # State-dependent (heteroscedastic) action std with an exploration floor.
            # A single global std (the default) over-specialises on easy terrain
            # (std -> ~0.18), which makes the policy so sharp that KL spikes floor the
            # adaptive LR at 1e-5; the frozen policy then can't adapt as the curriculum
            # climbs and de-learns irreversibly. A per-state std lets the policy stay
            # confident where it has mastered terrain and uncertain where it hasn't, and
            # the std_range floor (0.3) keeps minimum exploration so it never specialises
            # into the LR-flooring regime. The 2.0 ceiling (above init_std) both prevents
            # a runaway std-head explosion and keeps the clamp off the init so the std
            # gradient flows from the start.
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
            # Zero entropy bonus: it is the de-learning driver. Near convergence the
            # advantages vanish, so the surrogate gradient dies and the entropy bonus —
            # however small — dominates and inflates the action std until the policy
            # diffuses back to noise (confirmed: the same task with entropy_coef>0 still
            # de-learns even on a *fixed* terrain distribution). Exploration is instead
            # supplied by the std_range floor (0.3) on the per-state std head.
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            desired_kl=0.02,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),
    )
