"""GeneLab extension entry point: registers the G1 velocity tracking task."""

from __future__ import annotations

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    register_env,
    register_robot,
    register_task,
)
from genelab_unitree.g1 import (
    G1RobotCfg,
    get_g1_robot_cfg,
    unitree_g1_ppo_runner_cfg,
    unitree_g1_velocity_env_cfg,
)

TASK_ID = "Genelab-Velocity-Flat-Unitree-G1-v0"
ROBOT_NAME = "unitree-g1"
ENV_NAME = "g1-velocity-flat-env"


def _build_env(play: bool = False):
    """Late-import the env class so importing this module doesn't pull Genesis."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_velocity_env_cfg(play=play))


class G1VelocityTask:
    """Trainable task wrapper. Delegates to ``genelab.rl.runner`` for play / train."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TASK_ID,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            env=unitree_g1_velocity_env_cfg(play=False),
            play_env=unitree_g1_velocity_env_cfg(play=True),
            agent=unitree_g1_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


def register() -> None:
    if ROBOT_NAME not in ROBOTS:
        register_robot(
            ROBOT_NAME,
            get_g1_robot_cfg,
            description="Unitree G1 (29-DoF humanoid) for Genesis.",
            cfg_type=G1RobotCfg,
        )
    if ENV_NAME not in ENVS:
        register_env(
            ENV_NAME,
            lambda: _build_env(play=False),
            description="Unitree G1 velocity tracking on flat ground.",
            cfg_type=type(None),
        )
    if TASK_ID not in TASKS:
        register_task(
            TASK_ID,
            G1VelocityTask,
            description="PPO velocity tracking for Unitree G1 (flat).",
            cfg_type=TaskCfg,
        )
