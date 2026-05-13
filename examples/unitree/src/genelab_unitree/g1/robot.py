"""Unitree G1 robot factory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from genelab.envs.manager_based_rl_env import RobotEntityCfg
from genelab_unitree.g1.constants import (
    G1_ACTION_SCALE,
    G1_DEFAULT_JOINT_POS,
    G1_FOOT_LINKS,
    G1_INIT_BASE_HEIGHT,
    G1_JOINT_KP,
    G1_JOINT_KV,
)

# .../examples/unitree/src/genelab_unitree/g1/robot.py → .../examples/unitree/assets/g1/g1.xml
G1_MJCF_PATH: Final = (Path(__file__).resolve().parents[3] / "assets" / "g1" / "g1.xml").resolve()


@dataclass
class G1RobotCfg:
    """User-facing G1 robot config. Wraps a ``RobotEntityCfg`` for the env."""

    mjcf_path: str = field(default_factory=lambda: str(G1_MJCF_PATH))
    init_height: float = G1_INIT_BASE_HEIGHT

    def to_entity_cfg(self) -> RobotEntityCfg:
        return RobotEntityCfg(
            mjcf_path=self.mjcf_path,
            init_pos=(0.0, 0.0, self.init_height),
            init_quat=(1.0, 0.0, 0.0, 0.0),
            default_joint_pos=dict(G1_DEFAULT_JOINT_POS),
            joint_kp=dict(G1_JOINT_KP),
            joint_kv=dict(G1_JOINT_KV),
            action_scale=dict(G1_ACTION_SCALE),
            foot_link_names=G1_FOOT_LINKS,
        )


def get_g1_robot_cfg() -> G1RobotCfg:
    """Return a fresh G1 robot config (mutate-safe per call)."""
    return G1RobotCfg()
