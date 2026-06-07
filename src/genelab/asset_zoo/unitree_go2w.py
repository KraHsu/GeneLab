"""Unitree Go2-W asset zoo entry — 16-DoF wheeled quadruped (MuJoCo Menagerie source).

Go2-W is a Go2 with a powered wheel at the end of each leg, so it has two control
paradigms at once:

* 12 **position-controlled** leg joints — three ImplicitPD groups (hip / thigh / calf)
  with Isaac Lab-style gains (``stiffness=25, damping=0.5``); the calf carries the higher
  ``effort_limit=45.43`` the Menagerie forcerange reports, hip / thigh share ``23.7``.
* 4 **velocity-controlled** wheel joints — a single :class:`ImplicitVelocityActuator` group
  (Genesis ``control_dofs_velocity``, ``effort_limit=15``). The wheels are continuous joints
  with no range; a position target would fight their spin, so the policy commands a wheel
  *velocity* via ``mdp.JointVelocityAction``.

``foot_link_names`` points at the ``*_wheel_link`` bodies — the wheels are the ground
contact (each carries a geom of class ``foot``), so downstream contact rewards anchor there.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg, ImplicitVelocityActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="go2w",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/unitree_go2w/unitree_go2w.tar.gz",
    md5="afc9fac24cee7e5eade87cad75740ea1",
    filename="unitree_go2w.tar.gz",
    archive_member="unitree_go2w/go2w.xml",
)

_LEG_REGEX: Final = r"(FR|FL|RR|RL)"


def UnitreeGo2WCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the Unitree Go2-W wheeled quadruped.

    Leg home pose matches the standard Go2 stance (hip=0, thigh=0.9, calf=-1.8); the wheels
    start at rest (continuous, velocity-controlled).
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.45),
        default_joint_pos={
            rf"{_LEG_REGEX}_hip_joint": 0.0,
            rf"{_LEG_REGEX}_thigh_joint": 0.9,
            rf"{_LEG_REGEX}_calf_joint": -1.8,
        },
        actuators={
            "hip": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_hip_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=23.7,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
            "thigh": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_thigh_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=23.7,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
            "calf": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_calf_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=45.43,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
            # Velocity-controlled wheels: kp=0 (no position hold), kv=damping tracks the
            # commanded wheel speed. ``damping`` is a starting gain — tune against rolling.
            "wheel": ImplicitVelocityActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_wheel_joint",),
                damping=0.5,
                effort_limit=15.0,
            ),
        },
        foot_link_names=("FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link"),
    )


register_robot(
    "go2w",
    UnitreeGo2WCfg,
    description=(
        "Unitree Go2-W 16-DoF wheeled quadruped (Menagerie source); 12 ImplicitPD leg joints "
        "+ 4 velocity-controlled wheels."
    ),
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info go2w",
        "genelab list robots",
    ],
)
