"""Robot registrations for the external example tasks."""

from dataclasses import dataclass
from pathlib import Path
import tempfile

from genelab.registry import ROBOTS, register_robot

from genelab_examples.rubiks.assets import write_mjcf
from genelab_examples.rubiks.config import RubiksRobotCfg
from genelab_examples.wuji_hand.assets import resolve_description_dir, resolve_mjcf_path
from genelab_examples.wuji_hand.config import WujiRobotCfg


@dataclass(frozen=True)
class RubiksRobot:
    cfg: RubiksRobotCfg

    @property
    def name(self) -> str:
        return "rubiks-cube"

    def write_asset(self, output: Path | None = None) -> Path:
        spec = self.cfg.spec()
        path = (
            output or self.cfg.asset_output or Path(tempfile.gettempdir()) / "rubiks_cube_3x3x3.xml"
        )
        return write_mjcf(path, spec)


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


def create_rubiks_robot(cfg: RubiksRobotCfg | None = None) -> RubiksRobot:
    return RubiksRobot(cfg or RubiksRobotCfg())


def create_wuji_hand_robot(cfg: WujiRobotCfg | None = None) -> WujiHandRobot:
    return WujiHandRobot(cfg or WujiRobotCfg())


def register() -> None:
    if "rubiks-cube" not in ROBOTS:
        register_robot(
            "rubiks-cube",
            lambda: create_rubiks_robot(),
            description="Example Rubik's-cube-style MJCF robot asset.",
            cfg_type=RubiksRobotCfg,
        )
    if "wuji-hand" not in ROBOTS:
        register_robot(
            "wuji-hand",
            lambda: create_wuji_hand_robot(),
            description="Example Wuji dexterous hand MJCF robot asset.",
            cfg_type=WujiRobotCfg,
        )
