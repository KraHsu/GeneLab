"""Configuration for the Rubik's cube example task."""

from dataclasses import dataclass, field
from pathlib import Path

from genelab.configs import ManagerBasedEnvCfg, SceneCfg

from genelab_examples.rubiks.assets import RubiksCubeSpec
from genelab_examples.rubiks.sim import ForceDrivenCubeConfig


@dataclass
class RubiksRobotCfg:
    cubie_size: float = 0.05
    gap: float = 0.0015
    welded: bool = False
    asset_output: Path | None = None
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.25)
    rho: float = 500.0
    friction: float = 1.2

    def spec(self) -> RubiksCubeSpec:
        return RubiksCubeSpec(cubie_size=self.cubie_size, gap=self.gap, weld_cubies=self.welded)


@dataclass
class RubiksInteractionCfg:
    interactive_force: bool = False
    mouse_spring: float = 20.0
    mouse_max_distance: float = 0.08
    mouse_max_force: float = 1.0
    mouse_max_torque: float = 0.02


@dataclass
class LegacyLayerTurnCfg:
    enabled: bool = False
    axis: str = "z"
    layer: str = "+"
    turns: int = 1
    steps: int = 30


@dataclass
class RubiksEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = field(default_factory=SceneCfg)
    robot: RubiksRobotCfg = field(default_factory=RubiksRobotCfg)
    interaction: RubiksInteractionCfg = field(default_factory=RubiksInteractionCfg)
    legacy_turn: LegacyLayerTurnCfg = field(default_factory=LegacyLayerTurnCfg)
    force_controller: ForceDrivenCubeConfig = field(default_factory=ForceDrivenCubeConfig)
