"""Double inverted-pendulum constants: MJCF path, joint gains, action scale."""

from pathlib import Path
from typing import Final

# .../double/constants.py → .../assets/double_inverted_pendulum.xml
DOUBLE_INVERTED_PENDULUM_MJCF: Final = (
    Path(__file__).resolve().parents[3] / "assets" / "double_inverted_pendulum.xml"
).resolve()

CART_JOINT: Final = "cart_slide"
POLE_1_JOINT: Final = "pole_1_hinge"
POLE_2_JOINT: Final = "pole_2_hinge"
POLE_HINGE_JOINTS: Final = (POLE_1_JOINT, POLE_2_JOINT)
POLE_2_LINK: Final = "pole_2"

JOINT_KP: Final[dict[str, float]] = {CART_JOINT: 100.0}
JOINT_KV: Final[dict[str, float]] = {CART_JOINT: 10.0}

CART_ACTION_SCALE: Final[dict[str, float]] = {CART_JOINT: 1.0}

DEFAULT_JOINT_POS: Final[dict[str, float]] = {
    CART_JOINT: 0.0,
    POLE_1_JOINT: 0.0,
    POLE_2_JOINT: 0.0,
}

CART_POSITION_LIMIT: Final = 2.4
POLE_1_ANGLE_LIMIT: Final = 0.5
POLE_2_ANGLE_LIMIT: Final = 0.7

INIT_BASE_HEIGHT: Final = 0.12
