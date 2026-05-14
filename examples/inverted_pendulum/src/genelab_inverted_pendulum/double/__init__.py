"""Double inverted-pendulum robot, env, and PPO configuration."""

from genelab_inverted_pendulum.double.constants import (
    CART_ACTION_SCALE,
    CART_JOINT,
    DEFAULT_JOINT_POS,
    DOUBLE_INVERTED_PENDULUM_MJCF,
    JOINT_KP,
    JOINT_KV,
    POLE_1_JOINT,
    POLE_2_JOINT,
    POLE_2_LINK,
    POLE_HINGE_JOINTS,
)
from genelab_inverted_pendulum.double.env_cfg import double_inverted_pendulum_env_cfg
from genelab_inverted_pendulum.double.ppo_cfg import double_inverted_pendulum_ppo_runner_cfg
from genelab_inverted_pendulum.double.robot import (
    DoubleInvertedPendulumRobotCfg,
    get_double_inverted_pendulum_robot_cfg,
)

__all__ = [
    "CART_ACTION_SCALE",
    "CART_JOINT",
    "DEFAULT_JOINT_POS",
    "DOUBLE_INVERTED_PENDULUM_MJCF",
    "DoubleInvertedPendulumRobotCfg",
    "JOINT_KP",
    "JOINT_KV",
    "POLE_1_JOINT",
    "POLE_2_JOINT",
    "POLE_2_LINK",
    "POLE_HINGE_JOINTS",
    "double_inverted_pendulum_env_cfg",
    "double_inverted_pendulum_ppo_runner_cfg",
    "get_double_inverted_pendulum_robot_cfg",
]
