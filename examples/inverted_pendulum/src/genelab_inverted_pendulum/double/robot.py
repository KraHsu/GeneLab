"""Double inverted-pendulum robot factory."""

from dataclasses import dataclass, field

from genelab.envs.manager_based_rl_env import RobotEntityCfg

from genelab_inverted_pendulum.double.constants import (
    ACTUATORS_CFG,
    DEFAULT_JOINT_POS,
    DOUBLE_INVERTED_PENDULUM_MJCF,
    INIT_BASE_HEIGHT,
)


@dataclass
class DoubleInvertedPendulumRobotCfg:
    """User-facing double-pendulum config. Wraps a ``RobotEntityCfg`` for the env."""

    mjcf_path: str = field(default_factory=lambda: str(DOUBLE_INVERTED_PENDULUM_MJCF))
    init_height: float = INIT_BASE_HEIGHT

    def to_entity_cfg(self) -> RobotEntityCfg:
        return RobotEntityCfg(
            mjcf_path=self.mjcf_path,
            init_pos=(0.0, 0.0, self.init_height),
            init_quat=(1.0, 0.0, 0.0, 0.0),
            default_joint_pos=dict(DEFAULT_JOINT_POS),
            actuators=dict(ACTUATORS_CFG),
        )


def get_double_inverted_pendulum_robot_cfg() -> DoubleInvertedPendulumRobotCfg:
    return DoubleInvertedPendulumRobotCfg()
