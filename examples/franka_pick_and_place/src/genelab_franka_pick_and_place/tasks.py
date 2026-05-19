"""GeneLab extension entry point: registers the Franka pick-and-place task."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    register_env,
    register_robot,
    register_task,
)

from genelab_franka_pick_and_place.env_cfg import franka_pick_and_place_env_cfg
from genelab_franka_pick_and_place.ppo_cfg import franka_pick_and_place_ppo_runner_cfg
from genelab_franka_pick_and_place.robot import (
    FrankaPickAndPlaceRobotCfg,
    get_franka_pick_and_place_robot_cfg,
)

TASK_ID = "GeneLab-Franka-Pick-And-Place-v0"
ROBOT_NAME = "franka-pick-and-place"
ENV_NAME = "franka-pick-and-place-env"


def _build_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(franka_pick_and_place_env_cfg(play=play))


class FrankaPickAndPlaceTask:
    """Trainable Franka pick-and-place task. Delegates to ``genelab.rl.runner``."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TASK_ID,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            env=franka_pick_and_place_env_cfg(play=False),
            play_env=franka_pick_and_place_env_cfg(play=True),
            agent=franka_pick_and_place_ppo_runner_cfg(),
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
            get_franka_pick_and_place_robot_cfg,
            description="Franka Panda arm + parallel gripper at the world origin (pick-and-place).",
            cfg_type=FrankaPickAndPlaceRobotCfg,
        )
    if ENV_NAME not in ENVS:
        register_env(
            ENV_NAME,
            lambda: _build_env(play=False),
            description="Franka pick-and-place on a flat plane with a 4 cm cube.",
            cfg_type=type(None),
        )
    if TASK_ID not in TASKS:
        register_task(
            TASK_ID,
            FrankaPickAndPlaceTask,
            description="PPO Franka Panda pick-and-place (panda-gym-style dense reward).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {TASK_ID} --vis",
                f"genelab play {TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab train {TASK_ID} --num-envs 2048 --max-iterations 500",
            ],
        )
