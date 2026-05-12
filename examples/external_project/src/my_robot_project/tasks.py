"""GeneLab registration hook for the downstream package."""

from genelab.configs import TaskCfg
from genelab.registry import register_env, register_robot, register_task
from my_robot_project.config import MyEnvCfg, MyRobotCfg
from my_robot_project.envs import MyPickPlaceEnv
from my_robot_project.robots import create_robot


class MyPickPlaceTask:
    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name="MyProject-PickPlace-v0",
            env_name="my-project-pick-place",
            robot_name="my-project-robot",
            env=MyEnvCfg(),
            trainable=False,
        )

    def play(self) -> None:
        if not isinstance(self.cfg.env, MyEnvCfg):
            raise TypeError("MyPickPlaceTask requires MyEnvCfg")
        MyPickPlaceEnv(self.cfg.env).play()

    def train(self) -> None:
        raise NotImplementedError("add your RL runner in my_robot_project")


def register() -> None:
    register_robot(
        "my-project-robot",
        create_robot,
        description="Robot provided by my_robot_project.",
        cfg_type=MyRobotCfg,
    )
    register_env(
        "my-project-pick-place",
        MyPickPlaceEnv,
        description="Pick-place environment provided by my_robot_project.",
        cfg_type=MyEnvCfg,
    )
    register_task(
        "MyProject-PickPlace-v0",
        MyPickPlaceTask,
        description="Pick-place task provided by my_robot_project.",
        cfg_type=TaskCfg,
    )
