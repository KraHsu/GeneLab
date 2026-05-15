"""GeneLab extension entry point: registers the G1 velocity tracking + motion imitation tasks."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    TASKS,
    register_env,
    register_task,
)
from genelab_unitree.g1 import (
    unitree_g1_ppo_runner_cfg,
    unitree_g1_tracking_env_cfg,
    unitree_g1_tracking_ppo_runner_cfg,
    unitree_g1_velocity_env_cfg,
)

VELOCITY_TASK_ID = "Genelab-Velocity-Flat-Unitree-G1-v0"
TRACKING_TASK_ID = "Genelab-Tracking-Flat-Unitree-G1-v0"
# Matches the name registered by `genelab.asset_zoo.unitree_g1`.
ROBOT_NAME = "g1"
VELOCITY_ENV_NAME = "g1-velocity-flat-env"
TRACKING_ENV_NAME = "g1-tracking-flat-env"


def _build_velocity_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_velocity_env_cfg(play=play))


def _build_tracking_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_tracking_env_cfg(play=play))


class G1VelocityTask:
    """Trainable task wrapper. Delegates to ``genelab.rl.runner`` for play / train."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=VELOCITY_TASK_ID,
            env_name=VELOCITY_ENV_NAME,
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


class G1TrackingTask:
    """Motion-imitation task wrapper.

    The clip path is not committed; pass ``--env.commands.motion.motion_file PATH`` (or override
    ``self.cfg.env.commands_cfg["motion"].motion_file``) before invoking play/train.
    """

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TRACKING_TASK_ID,
            env_name=TRACKING_ENV_NAME,
            robot_name=ROBOT_NAME,
            env=unitree_g1_tracking_env_cfg(play=False),
            play_env=unitree_g1_tracking_env_cfg(play=True),
            agent=unitree_g1_tracking_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, agent="zero")

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


def register() -> None:
    # The G1 robot itself is registered by `genelab.asset_zoo.unitree_g1`; no robot registration here.
    if VELOCITY_ENV_NAME not in ENVS:
        register_env(
            VELOCITY_ENV_NAME,
            lambda: _build_velocity_env(play=False),
            description="Unitree G1 velocity tracking on flat ground.",
            cfg_type=type(None),
        )
    if TRACKING_ENV_NAME not in ENVS:
        register_env(
            TRACKING_ENV_NAME,
            lambda: _build_tracking_env(play=False),
            description="Unitree G1 motion imitation (BeyondMimic-style) on flat ground.",
            cfg_type=type(None),
        )
    if VELOCITY_TASK_ID not in TASKS:
        register_task(
            VELOCITY_TASK_ID,
            G1VelocityTask,
            description="PPO velocity tracking for Unitree G1 (flat).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {VELOCITY_TASK_ID}",
                f"genelab play {VELOCITY_TASK_ID} --agent random",
                f"genelab play {VELOCITY_TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab train {VELOCITY_TASK_ID} --num_envs 4096",
                f"genelab train {VELOCITY_TASK_ID} --num_envs 4096 --gpus 2",
            ],
        )
    if TRACKING_TASK_ID not in TASKS:
        register_task(
            TRACKING_TASK_ID,
            G1TrackingTask,
            description=(
                "PPO motion imitation for Unitree G1 on flat ground. "
                "Requires a motion clip; see examples."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {TRACKING_TASK_ID} --env.commands.motion.motion_file PATH/clip.npy",
                f"genelab train {TRACKING_TASK_ID} "
                "--env.commands.motion.motion_file PATH/clip.npy --num_envs 4096",
                f"genelab train {TRACKING_TASK_ID} "
                "--env.commands.motion.motion_file PATH/clip.npy --num_envs 4096 --gpus 2",
            ],
        )
