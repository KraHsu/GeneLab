"""Robot registrations for the Wuji hand example tasks."""

from dataclasses import dataclass
from pathlib import Path

from genelab.registry import ROBOTS, register_robot

from genelab_wuji.wuji_hand.assets import resolve_description_dir, resolve_mjcf_path
from genelab_wuji.wuji_hand.config import WujiRobotCfg


@dataclass(frozen=True)
class WujiHandRobot:
    cfg: WujiRobotCfg

    @property
    def name(self) -> str:
        return "wuji-hand"

    @property
    def desc_dir(self) -> Path:
        return resolve_description_dir(self.cfg.desc_dir)

    @property
    def mjcf_path(self) -> Path:
        return resolve_mjcf_path(self.cfg.desc_dir, self.cfg.side)


def create_wuji_hand_robot(cfg: WujiRobotCfg | None = None) -> WujiHandRobot:
    return WujiHandRobot(cfg or WujiRobotCfg())


def register() -> None:
    if "wuji-hand" not in ROBOTS:
        register_robot(
            "wuji-hand",
            lambda: create_wuji_hand_robot(),
            description="Example Wuji dexterous hand MJCF robot asset.",
            cfg_type=WujiRobotCfg,
        )
