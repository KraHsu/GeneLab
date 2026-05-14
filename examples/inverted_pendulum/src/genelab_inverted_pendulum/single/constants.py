"""Single inverted-pendulum constants: MJCF path, actuator groups, default pose."""

from pathlib import Path
from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg

# .../single/constants.py → .../assets/inverted_pendulum.xml
INVERTED_PENDULUM_MJCF: Final = (
    Path(__file__).resolve().parents[3] / "assets" / "inverted_pendulum.xml"
).resolve()

CART_JOINT: Final = "cart_slide"
POLE_JOINT: Final = "pole_hinge"
POLE_LINK: Final = "pole"

# Underactuated: only the cart slide receives PD; the pole hinge is passive (zero gains).
CART_ACTUATOR_CFG: Final = ImplicitPDActuatorCfg(
    target_names_expr=(CART_JOINT,),
    stiffness=80.0,
    damping=8.0,
    action_scale=1.0,
)
POLE_ACTUATOR_CFG: Final = ImplicitPDActuatorCfg(
    target_names_expr=(POLE_JOINT,),
    stiffness=0.0,
    damping=0.0,
    action_scale=0.0,
)
ACTUATORS_CFG: Final = {"cart": CART_ACTUATOR_CFG, "pole": POLE_ACTUATOR_CFG}

DEFAULT_JOINT_POS: Final[dict[str, float]] = {CART_JOINT: 0.0, POLE_JOINT: 0.0}

# Termination limits.
CART_POSITION_LIMIT: Final = 2.4
POLE_ANGLE_LIMIT: Final = 0.4  # ~23°

INIT_BASE_HEIGHT: Final = 0.12
