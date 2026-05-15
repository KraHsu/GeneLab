"""Double inverted-pendulum constants: MJCF path, actuator groups, default pose."""

from pathlib import Path
from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg

# .../double/constants.py → .../assets/double_inverted_pendulum.xml
DOUBLE_INVERTED_PENDULUM_MJCF: Final = (
    Path(__file__).resolve().parents[3] / "assets" / "double_inverted_pendulum.xml"
).resolve()

CART_JOINT: Final = "cart_slide"
POLE_1_JOINT: Final = "pole_1_hinge"
POLE_2_JOINT: Final = "pole_2_hinge"
POLE_HINGE_JOINTS: Final = (POLE_1_JOINT, POLE_2_JOINT)
POLE_2_LINK: Final = "pole_2"

CART_ACTUATOR_CFG: Final = ImplicitPDActuatorCfg(
    target_names_expr=(CART_JOINT,),
    stiffness=100.0,
    damping=10.0,
    action_scale=1.0,
)
POLE_ACTUATOR_CFG: Final = ImplicitPDActuatorCfg(
    target_names_expr=(POLE_1_JOINT, POLE_2_JOINT),
    stiffness=0.0,
    damping=0.0,
    action_scale=0.0,
)
ACTUATORS_CFG: Final = {"cart": CART_ACTUATOR_CFG, "poles": POLE_ACTUATOR_CFG}

DEFAULT_JOINT_POS: Final[dict[str, float]] = {
    CART_JOINT: 0.0,
    POLE_1_JOINT: 0.0,
    POLE_2_JOINT: 0.0,
}

CART_POSITION_LIMIT: Final = 2.4
POLE_1_ANGLE_LIMIT: Final = 0.5
POLE_2_ANGLE_LIMIT: Final = 0.7

INIT_BASE_HEIGHT: Final = 0.12
