"""Constants shared across robot / env / reward cfg for the pick-and-place task."""

from typing import Final

# Franka link / joint identifiers (Menagerie naming).
EE_LINK: Final = "hand"
ARM_JOINTS_REGEX: Final = r"^joint[1-7]$"
FINGER_JOINTS_REGEX: Final = r"^finger_joint.*$"

# Cube / object geometry.
CUBE_SIZE: Final = 0.04  # 4 cm cube — matches panda-gym's `object_size`.
CUBE_HALF: Final = CUBE_SIZE / 2.0

# Cube spawn distribution (uniform offset around the nominal table-front position).
CUBE_NOMINAL_XY: Final[tuple[float, float]] = (0.5, 0.0)
CUBE_XY_RANGE: Final[tuple[float, float]] = (-0.15, 0.15)

# Goal sampling (panda-gym parity): xy uniform around (0.5, 0), z either on the
# ground (probability 1 - GOAL_AIR_PROB) or uniform in [0, GOAL_Z_MAX].
GOAL_XY_RANGE: Final[tuple[float, float]] = (-0.15, 0.15)
GOAL_Z_MAX: Final = 0.2
GOAL_AIR_PROB: Final = 0.7

# Success threshold (panda-gym `distance_threshold = 0.05`).
DISTANCE_THRESHOLD: Final = 0.05

# Safety: terminate the episode if the cube falls below this z (knocked off-world).
CUBE_DROP_Z: Final = -0.1

# Entity name used inside InteractiveSceneCfg.entities (also referenced from MDP terms).
CUBE_ENTITY_NAME: Final = "cube"
