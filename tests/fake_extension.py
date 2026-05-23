"""Fake downstream GeneLab extension used by CLI tests."""

from dataclasses import dataclass, field

from genelab.configs import ManagerBasedEnvCfg, SimulationCfg, TaskCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.registry import register_env, register_robot, register_task


@dataclass
class FakeRobotCfg:
    family: str = "external"


@dataclass
class FakeEnvCfg(ManagerBasedEnvCfg):
    simulation: SimulationCfg = field(default_factory=lambda: SimulationCfg(steps=0))
    label: str = "fake-extension"


@dataclass
class FakeRlEnvCfg(ManagerBasedRlEnvCfg):
    """RL-capable env cfg (subclasses the RL base) used to test that RL play options
    route through the ``play_task`` helper — the counterpart to ``FakeEnvCfg``, which is
    a non-RL scene demo."""

    simulation: SimulationCfg = field(default_factory=lambda: SimulationCfg(steps=0))
    label: str = "fake-extension-rl"


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
        print(f"played {self.cfg.name} for {self.cfg.env.simulation.steps} steps")

    def train(self) -> None:
        raise NotImplementedError("training is not implemented for fake extension")


class FakeRlTask:
    """An RL-capable task: its env cfg is a ``ManagerBasedRlEnvCfg``, so RL play options
    (``--agent`` / ``--checkpoint`` / …) must route through the ``play_task`` helper."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name="External-Fake-RL-Task-v0",
            env_name="fake-extension-rl-env",
            robot_name="fake-extension-robot",
            env=FakeRlEnvCfg(),
            trainable=False,
        )

    def play(self) -> None:
        raise AssertionError("RL-capable task must route through play_task, not task.play()")

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
        examples=[
            "genelab play External-Fake-Task-v0",
            "genelab play External-Fake-Task-v0 --steps 7",
        ],
    )
    register_task(
        "External-Fake-RL-Task-v0",
        FakeRlTask,
        description="RL-capable task from a fake external package.",
        cfg_type=TaskCfg,
    )
