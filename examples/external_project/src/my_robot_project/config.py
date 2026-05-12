"""Configs owned by the downstream robotics project."""

from dataclasses import dataclass, field

from genelab.configs import ManagerBasedEnvCfg, SceneCfg


@dataclass
class MyRobotCfg:
    usd_path: str = "assets/my_robot.usd"


@dataclass
class MyEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(steps=128))
    robot: MyRobotCfg = field(default_factory=MyRobotCfg)
