"""Single inverted-pendulum constants: MJCF path, joint gains, action scale."""

from pathlib import Path
from typing import Final

# .../single/constants.py → .../assets/inverted_pendulum.xml
INVERTED_PENDULUM_MJCF: Final = (
    Path(__file__).resolve().parents[3] / "assets" / "inverted_pendulum.xml"
).resolve()

CART_JOINT: Final = "cart_slide"
POLE_JOINT: Final = "pole_hinge"
POLE_LINK: Final = "pole"

# Underactuated: only the cart slide receives PD. The pole hinge defaults to kp=0/kv=0.
JOINT_KP: Final[dict[str, float]] = {CART_JOINT: 80.0}
JOINT_KV: Final[dict[str, float]] = {CART_JOINT: 8.0}

# JointPositionAction scale — policy action ∈ [-1, 1] maps to ±1.0 m setpoint about the cart origin.
CART_ACTION_SCALE: Final[dict[str, float]] = {CART_JOINT: 1.0}

DEFAULT_JOINT_POS: Final[dict[str, float]] = {CART_JOINT: 0.0, POLE_JOINT: 0.0}

# Termination limits.
CART_POSITION_LIMIT: Final = 2.4
POLE_ANGLE_LIMIT: Final = 0.4  # ~23°

INIT_BASE_HEIGHT: Final = 0.12
