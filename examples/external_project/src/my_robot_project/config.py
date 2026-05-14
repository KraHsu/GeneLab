"""Configs owned by the downstream robotics project."""

from dataclasses import dataclass, field

from genelab.configs import ManagerBasedEnvCfg, SimulationCfg


@dataclass
class MyRobotCfg:
    usd_path: str = "assets/my_robot.usd"


@dataclass
class MyEnvCfg(ManagerBasedEnvCfg):
    simulation: SimulationCfg = field(default_factory=lambda: SimulationCfg(steps=128))
    robot: MyRobotCfg = field(default_factory=MyRobotCfg)
