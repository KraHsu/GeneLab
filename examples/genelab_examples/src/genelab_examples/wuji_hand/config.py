"""Configuration for the Wuji hand example task."""

from dataclasses import dataclass, field
from pathlib import Path

from genelab.configs import ManagerBasedEnvCfg, SceneCfg

from genelab_examples.wuji_hand.assets import DEFAULT_DESC_DIR, DEFAULT_TRAJECTORY


@dataclass
class WujiRobotCfg:
    side: str = "right"
    desc_dir: Path = DEFAULT_DESC_DIR
    trajectory: Path = DEFAULT_TRAJECTORY


@dataclass
class WujiEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(steps=0, dt=0.01, substeps=1))
    robot: WujiRobotCfg = field(default_factory=WujiRobotCfg)
    reset_interval: int = 500
