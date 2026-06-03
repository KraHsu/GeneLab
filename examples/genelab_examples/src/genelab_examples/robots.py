"""Robot registrations for the external example tasks."""

from dataclasses import dataclass
from pathlib import Path
import tempfile

from genelab.registry import ROBOTS, register_robot

from genelab_examples.rubiks.assets import write_mjcf
from genelab_examples.rubiks.config import RubiksRobotCfg


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


def create_rubiks_robot(cfg: RubiksRobotCfg | None = None) -> RubiksRobot:
    return RubiksRobot(cfg or RubiksRobotCfg())


def register() -> None:
    if "rubiks-cube" not in ROBOTS:
        register_robot(
            "rubiks-cube",
            lambda: create_rubiks_robot(),
            description="Example Rubik's-cube-style MJCF robot asset.",
            cfg_type=RubiksRobotCfg,
        )
