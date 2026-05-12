"""Robot definitions owned by the downstream package."""

from dataclasses import dataclass

from my_robot_project.config import MyRobotCfg


@dataclass
class MyRobot:
    cfg: MyRobotCfg


def create_robot(cfg: MyRobotCfg | None = None) -> MyRobot:
    return MyRobot(cfg or MyRobotCfg())
