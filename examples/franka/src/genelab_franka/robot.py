"""Robot factory for the pick-and-place task — wraps the asset-zoo Franka."""

from dataclasses import dataclass, field

from genelab.asset_zoo.franka import FrankaPandaCfg
from genelab.entity import ArticulationCfg

# panda-gym ``PandaPickAndPlace`` neutral arm pose (radians). The Cartesian
# variant locks the EE orientation to whatever pose the arm resets into, so this
# also fixes the gripper's downward orientation for differential IK. Differs
# from the Menagerie ``home`` keyframe baked into ``FrankaPandaCfg``.
PANDA_GYM_NEUTRAL_POSE: dict[str, float] = {
    "joint1": 0.0,
    "joint2": 0.41,
    "joint3": 0.0,
    "joint4": -1.85,
    "joint5": 0.0,
    "joint6": 2.26,
    "joint7": 0.79,
    "finger_joint.*": 0.04,
}


@dataclass
class FrankaPickAndPlaceRobotCfg:
    """User-facing config; ``to_entity_cfg()`` returns the underlying ``ArticulationCfg``.

    Keeps the wrapping pattern used by the inverted-pendulum example so the registry's
    ``cfg_type`` slot stays meaningful. ``requires_jac_and_ik`` toggles the Genesis
    morph flag needed by Cartesian (differential-IK) action terms — the
    joint-position variant leaves it ``False`` to avoid paying for unused buffers.
    ``default_joint_pos`` overrides the Menagerie ``home`` keyframe when set (the
    Cartesian variant uses the panda-gym neutral pose).
    """

    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    requires_jac_and_ik: bool = False
    default_joint_pos: dict[str, float] | None = field(default=None)

    def to_entity_cfg(self) -> ArticulationCfg:
        cfg = FrankaPandaCfg()
        cfg.init_pos = (self.base_pos[0], self.base_pos[1], self.base_pos[2])
        cfg.requires_jac_and_ik = self.requires_jac_and_ik
        if self.default_joint_pos is not None:
            cfg.default_joint_pos = dict(self.default_joint_pos)
        return cfg


def get_franka_pick_and_place_robot_cfg(
    requires_jac_and_ik: bool = False,
    default_joint_pos: dict[str, float] | None = None,
) -> FrankaPickAndPlaceRobotCfg:
    return FrankaPickAndPlaceRobotCfg(
        requires_jac_and_ik=requires_jac_and_ik,
        default_joint_pos=default_joint_pos,
    )
