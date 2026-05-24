"""skrl PPO agent config for the single inverted-pendulum task.

Mirrors :func:`inverted_pendulum_ppo_runner_cfg` (the rsl_rl PPO recipe) so the
same env can be driven by the skrl backend — the backend is chosen purely by the
agent-cfg type. ``SkrlAgentCfg`` is a plain dataclass that imports nothing from
skrl, so registering this task never requires skrl to be installed.
"""

from genelab.rl import SkrlAgentCfg, SkrlExperimentCfg, SkrlModelCfg


def inverted_pendulum_skrl_agent_cfg() -> SkrlAgentCfg:
    return SkrlAgentCfg(
        algorithm="PPO",
        seed=42,
        # skrl trains in environment timesteps (``--max-iterations`` maps onto this).
        # 4800 timesteps / 24-step rollout buffer = 200 PPO updates, comparable to
        # the rsl_rl recipe's iteration budget.
        timesteps=4800,
        rollouts=24,
        learning_rate=1.0e-3,
        discount_factor=0.99,
        learning_epochs=5,
        mini_batches=4,
        gae_lambda=0.95,
        ratio_clip=0.2,
        entropy_loss_scale=0.005,
        grad_norm_clip=1.0,
        models=SkrlModelCfg(hidden_dims=(128, 128), activation="elu"),
        experiment=SkrlExperimentCfg(experiment_name="inverted_pendulum_skrl"),
        obs_group="policy",
        state_group="critic",
    )
