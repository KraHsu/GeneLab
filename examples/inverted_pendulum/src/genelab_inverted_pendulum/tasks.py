"""GeneLab extension entry point: registers single + double inverted-pendulum tasks."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    register_env,
    register_robot,
    register_task,
)

from genelab_inverted_pendulum.double import (
    DoubleInvertedPendulumRobotCfg,
    double_inverted_pendulum_env_cfg,
    double_inverted_pendulum_ppo_runner_cfg,
    get_double_inverted_pendulum_robot_cfg,
)
from genelab_inverted_pendulum.single import (
    InvertedPendulumRobotCfg,
    get_inverted_pendulum_robot_cfg,
    inverted_pendulum_env_cfg,
    inverted_pendulum_ppo_runner_cfg,
)

INVERTED_PENDULUM_TASK_ID = "GeneLab-Inverted-Pendulum-v0"
DOUBLE_PENDULUM_TASK_ID = "GeneLab-Double-Inverted-Pendulum-v0"
ROBOT_NAME_SINGLE = "inverted-pendulum"
ROBOT_NAME_DOUBLE = "double-inverted-pendulum"
ENV_NAME_SINGLE = "inverted-pendulum-env"
ENV_NAME_DOUBLE = "double-inverted-pendulum-env"


def _build_single_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(inverted_pendulum_env_cfg(play=play))


def _build_double_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(double_inverted_pendulum_env_cfg(play=play))


class InvertedPendulumTask:
    """Trainable single inverted-pendulum task. Delegates to ``genelab.rl.runner``."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=INVERTED_PENDULUM_TASK_ID,
            env_name=ENV_NAME_SINGLE,
            robot_name=ROBOT_NAME_SINGLE,
            env=inverted_pendulum_env_cfg(play=False),
            play_env=inverted_pendulum_env_cfg(play=True),
            agent=inverted_pendulum_ppo_runner_cfg(),
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


class DoubleInvertedPendulumTask:
    """Trainable double inverted-pendulum task."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=DOUBLE_PENDULUM_TASK_ID,
            env_name=ENV_NAME_DOUBLE,
            robot_name=ROBOT_NAME_DOUBLE,
            env=double_inverted_pendulum_env_cfg(play=False),
            play_env=double_inverted_pendulum_env_cfg(play=True),
            agent=double_inverted_pendulum_ppo_runner_cfg(),
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
    if ROBOT_NAME_SINGLE not in ROBOTS:
        register_robot(
            ROBOT_NAME_SINGLE,
            get_inverted_pendulum_robot_cfg,
            description="Cart with a single hinge-mounted inverted pole.",
            cfg_type=InvertedPendulumRobotCfg,
        )
    if ROBOT_NAME_DOUBLE not in ROBOTS:
        register_robot(
            ROBOT_NAME_DOUBLE,
            get_double_inverted_pendulum_robot_cfg,
            description="Cart with two serially linked inverted poles.",
            cfg_type=DoubleInvertedPendulumRobotCfg,
        )
    if ENV_NAME_SINGLE not in ENVS:
        register_env(
            ENV_NAME_SINGLE,
            lambda: _build_single_env(play=False),
            description="Single inverted-pendulum balancing on a flat plane.",
            cfg_type=type(None),
        )
    if ENV_NAME_DOUBLE not in ENVS:
        register_env(
            ENV_NAME_DOUBLE,
            lambda: _build_double_env(play=False),
            description="Double inverted-pendulum balancing on a flat plane.",
            cfg_type=type(None),
        )
    if INVERTED_PENDULUM_TASK_ID not in TASKS:
        register_task(
            INVERTED_PENDULUM_TASK_ID,
            InvertedPendulumTask,
            description="PPO inverted-pendulum balancing (single pole).",
            cfg_type=TaskCfg,
        )
    if DOUBLE_PENDULUM_TASK_ID not in TASKS:
        register_task(
            DOUBLE_PENDULUM_TASK_ID,
            DoubleInvertedPendulumTask,
            description="PPO double-inverted-pendulum balancing (two stacked poles).",
            cfg_type=TaskCfg,
        )
