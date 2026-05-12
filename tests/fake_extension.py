"""Fake downstream GeneLab extension used by CLI tests."""

from dataclasses import dataclass, field

from genelab.configs import ManagerBasedEnvCfg, SceneCfg, TaskCfg
from genelab.registry import register_env, register_robot, register_task


@dataclass
class FakeRobotCfg:
    family: str = "external"


@dataclass
class FakeEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(steps=0))
    label: str = "fake-extension"


@dataclass
class FakeRobot:
    cfg: FakeRobotCfg = field(default_factory=FakeRobotCfg)


class FakeTask:
    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name="External-Fake-Task-v0",
            env_name="fake-extension-env",
            robot_name="fake-extension-robot",
            env=FakeEnvCfg(),
            trainable=False,
        )

    def play(self) -> None:
        if not isinstance(self.cfg.env, FakeEnvCfg):
            raise TypeError("FakeTask requires FakeEnvCfg")
        print(f"played {self.cfg.name} for {self.cfg.env.scene.steps} steps")

    def train(self) -> None:
        raise NotImplementedError("training is not implemented for fake extension")


def register() -> None:
    register_robot(
        "fake-extension-robot",
        FakeRobot,
        description="Robot from a fake external package.",
        cfg_type=FakeRobotCfg,
    )
    register_env(
        "fake-extension-env",
        FakeEnvCfg,
        description="Environment from a fake external package.",
        cfg_type=FakeEnvCfg,
    )
    register_task(
        "External-Fake-Task-v0",
        FakeTask,
        description="Task from a fake external package.",
        cfg_type=TaskCfg,
    )
