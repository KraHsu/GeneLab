"""Stable-Baselines3 agent configs for the Franka pick-and-place task.

Two configs are exposed:

* :func:`franka_pick_and_place_sb3_cfg` — SB3 PPO, the direct counterpart of
  ``ppo_cfg.py`` (RSL-RL) and ``skrl_cfg.py`` (skrl).
* :func:`franka_pick_and_place_sb3_her_cfg` — SB3 SAC + Hindsight Experience
  Replay, the panda-gym-style goal-conditioned setup.

SB3 trains in environment timesteps rather than learning iterations.
"""

import numpy as np

from genelab.rl import Sb3AgentCfg, Sb3ExperimentCfg, Sb3HerCfg, Sb3PolicyCfg

from genelab_franka_pick_and_place.constants import DISTANCE_THRESHOLD


def franka_pick_and_place_her_compute_reward(
    achieved_goal: np.ndarray, desired_goal: np.ndarray, info: object
) -> np.ndarray:
    """panda-gym-style **sparse** goal reward for HER goal relabelling.

    ``-1`` while the cube is farther than ``DISTANCE_THRESHOLD`` from the goal,
    ``0`` once it is within threshold. Vectorized over a batch of goal pairs
    (``HerReplayBuffer`` passes ``(batch, 3)`` numpy arrays)."""
    distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
    return -(distance > DISTANCE_THRESHOLD).astype(np.float32)


def franka_pick_and_place_sb3_cfg() -> Sb3AgentCfg:
    """SB3 PPO config — counterpart of the RSL-RL / skrl Franka configs."""
    return Sb3AgentCfg(
        algorithm="PPO",
        seed=42,
        total_timesteps=500_000,
        learning_rate=3.0e-4,
        discount_factor=0.99,
        n_steps=32,
        batch_size=256,
        n_epochs=5,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy=Sb3PolicyCfg(net_arch=(256, 256, 128), activation="elu"),
        experiment=Sb3ExperimentCfg(
            experiment_name="franka_pick_and_place_sb3",
            logger="tensorboard",
        ),
    )


def franka_pick_and_place_sb3_her_cfg() -> Sb3AgentCfg:
    """SB3 SAC + HER config — goal-conditioned pick-and-place (panda-gym style).

    ``HerReplayBuffer`` cannot sample until a full episode has finished in every
    env, so ``learning_starts`` must cover ``num_envs x episode_length``. The SB3
    backend raises ``learning_starts`` to that floor automatically, so the value
    set here only matters when it already exceeds it."""
    return Sb3AgentCfg(
        algorithm="SAC",
        seed=42,
        total_timesteps=500_000,
        learning_rate=3.0e-4,
        discount_factor=0.99,
        batch_size=256,
        buffer_size=200_000,
        tau=0.005,
        learning_starts=4000,
        train_freq=1,
        policy=Sb3PolicyCfg(net_arch=(256, 256, 128), activation="elu"),
        experiment=Sb3ExperimentCfg(
            experiment_name="franka_pick_and_place_sb3_her",
            logger="tensorboard",
        ),
        her=Sb3HerCfg(
            enabled=True,
            obs_group="policy",
            achieved_goal_group="achieved_goal",
            desired_goal_group="desired_goal",
            n_sampled_goal=4,
            goal_selection_strategy="future",
            compute_reward=franka_pick_and_place_her_compute_reward,
        ),
    )
