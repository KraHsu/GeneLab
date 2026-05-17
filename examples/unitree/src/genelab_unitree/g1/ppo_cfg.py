"""PPO config for the Unitree G1 velocity tracking task.

Originally a 1:1 mirror of ``mjlab.tasks.velocity.config.g1.rl_cfg.unitree_g1_ppo_runner_cfg``.
Two knobs now diverge from mjlab to avoid the early entropy collapse observed
on Genesis (logs: ``Policy/mean_std`` dropped 1.0 → 0.17 within 2k iters,
``Loss/entropy`` went negative, ``track_lin_vel`` plateaued near 0.3 and the
policy never learned to step):

* ``init_std=0.5`` (mjlab: 1.0) — with 29 actuated DoFs, σ=1.0 produces a
  ~σ²·29 action-norm² blowup on the first rollouts, triggering huge KL spikes.
* ``desired_kl=0.02`` (mjlab: 0.01) — paired with the adaptive lr schedule the
  tighter target was crushing the policy std before any locomotion gradient
  could form.

Everything else (network width / depth, entropy_coef, num_steps_per_env,
max_iterations, save_interval, learning rate) still matches mjlab's reference.

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
                "init_std": 0.5,
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
            desired_kl=0.02,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),
    )
