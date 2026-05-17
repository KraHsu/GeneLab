"""PPO config for the Unitree G1 velocity tracking task.

1:1 mirror of ``mjlab.tasks.velocity.config.g1.rl_cfg.unitree_g1_ppo_runner_cfg``.
GeneLab adds two small extras mjlab leaves to defaults: a fixed RNG seed (``42``)
and the explicit ``logger="tensorboard"`` selection. Everything else (network
width / depth, init_std, entropy_coef, schedule, num_steps_per_env,
max_iterations, save_interval) matches mjlab's reference.

An earlier divergence (``init_std=0.5`` / ``desired_kl=0.02``) was introduced to
mask an early entropy collapse on Genesis, but the underlying cause turned out
to be a stale joint-state read in the substep PD path — ``Articulation`` only
refreshed its joint_pos / joint_vel cache once per env step, so the Python-side
PD ran open-loop inside the decimation window and the resulting joint
oscillation made every random rollout fall in < 0.5 s. With that bug fixed the
mjlab std/kl values train cleanly; reverting the divergence also restores the
exploration the gait-shaping rewards need to drive feet off the ground.

mjlab does **not** set ``clip_actions``; we leave GeneLab's default (``None``)
in place so the policy output passes through without pre-env clipping —
the joint position action already routes through the actuator group's scale.
"""

from genelab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=30_000,
        save_interval=50,
        experiment_name="g1_velocity",
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
