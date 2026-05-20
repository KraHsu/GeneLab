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

    ``HerReplayBuffer`` works in whole episodes: both ``learning_starts`` and
    ``buffer_size`` must cover ``num_envs x episode_length``. The SB3 backend
    raises either to that floor automatically, so the values here only bind when
    they already exceed it. HER does not benefit from massive parallelism — a
    modest ``--num-envs`` (≤ ~64) keeps ``buffer_size`` worth many episodes per
    env; with thousands of envs the buffer holds barely one episode each."""
    return Sb3AgentCfg(
        algorithm="SAC",
        seed=42,
        total_timesteps=1_000_000,
        learning_rate=1.0e-3,
        discount_factor=0.95,
        batch_size=2048,
        buffer_size=1_000_000,
        tau=0.05,
        learning_starts=4000,
        train_freq=1,
        # SAC's default auto-tuned alpha collapsed to ~1e-4 within the first 10%
        # of training in the 2026-05-20 run, killing exploration; pin alpha to
        # the panda-gym-tested fixed value instead. (Sb3AgentCfg.ent_coef is
        # consumed by PPO/A2C only; SAC accepts this through extra_kwargs.)
        extra_kwargs={"ent_coef": 0.1},
        policy=Sb3PolicyCfg(net_arch=(512, 512, 512), activation="relu"),
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
