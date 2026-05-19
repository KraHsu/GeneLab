"""skrl PPO agent config for the Franka pick-and-place task.

Mirrors ``ppo_cfg.py`` (RSL-RL) but drives the skrl backend. skrl trains in
environment timesteps rather than learning iterations: with ``--num-envs 2048``
the default ``timesteps`` below is ~24M env-steps, comparable to the RSL-RL run
(500 iterations x 24 steps).
"""

from genelab.rl import SkrlAgentCfg, SkrlExperimentCfg, SkrlModelCfg


def franka_pick_and_place_skrl_cfg() -> SkrlAgentCfg:
    return SkrlAgentCfg(
        algorithm="PPO",
        seed=42,
        timesteps=12_000,
        rollouts=24,
        learning_rate=3.0e-4,
        discount_factor=0.99,
        learning_epochs=5,
        mini_batches=4,
        gae_lambda=0.95,
        ratio_clip=0.2,
        entropy_loss_scale=0.005,
        grad_norm_clip=1.0,
        models=SkrlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
        ),
        experiment=SkrlExperimentCfg(
            experiment_name="franka_pick_and_place_skrl",
            logger="tensorboard",
        ),
    )
