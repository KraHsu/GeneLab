"""Robot factory for the pick-and-place task — wraps the asset-zoo Franka."""

from dataclasses import dataclass

from genelab.asset_zoo.franka import FrankaPandaCfg
from genelab.entity import ArticulationCfg


@dataclass
class FrankaPickAndPlaceRobotCfg:
    """User-facing config; ``to_entity_cfg()`` returns the underlying ``ArticulationCfg``.

    Keeps the wrapping pattern used by the inverted-pendulum example so the registry's
    ``cfg_type`` slot stays meaningful. ``requires_jac_and_ik`` toggles the Genesis
    morph flag needed by Cartesian (differential-IK) action terms — the
    joint-position variant leaves it ``False`` to avoid paying for unused buffers.
    """

    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    requires_jac_and_ik: bool = False

    def to_entity_cfg(self) -> ArticulationCfg:
        cfg = FrankaPandaCfg()
        cfg.init_pos = (self.base_pos[0], self.base_pos[1], self.base_pos[2])
        cfg.requires_jac_and_ik = self.requires_jac_and_ik
        return cfg


def get_franka_pick_and_place_robot_cfg(
    requires_jac_and_ik: bool = False,
) -> FrankaPickAndPlaceRobotCfg:
    return FrankaPickAndPlaceRobotCfg(requires_jac_and_ik=requires_jac_and_ik)
