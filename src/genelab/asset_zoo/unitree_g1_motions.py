"""Unitree G1 motion-clip asset specs.

Demo clips for the BeyondMimic-style motion-imitation task in
``examples/unitree``. Each entry mirrors the ``AssetSpec`` pattern used for
robot MJCFs: an md5-pinned download from the ``genelab-assets`` repo, fetched
lazily on first call. NPZ files follow the schema
:class:`genelab.mdp.commands.motion_command.MotionLoader` consumes.

The bundled `dance1_subject2.npz` is derived from the Unitree-retargeted
LAFAN1 dataset and inherits CC BY-NC-ND 4.0 — non-commercial research use
only. See ``unitree_g1/motions/LICENSE.NOTICE`` in the assets repo.

The body / joint axes inside the NPZ follow mjlab's MJCF DFS traversal order
(see :data:`G1_MJLAB_BODY_NAMES` / :data:`G1_MJLAB_JOINT_NAMES`); Genesis uses
its own traversal so the env config must hand these orderings to
``MotionCommandCfg`` for a per-axis permutation.
"""

from pathlib import Path
from typing import Final

from genelab.utils.download import AssetSpec, fetch_asset


G1_MJLAB_BODY_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

G1_MJLAB_JOINT_NAMES: Final[tuple[str, ...]] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_DANCE1_SUBJECT2: Final = AssetSpec(
    name="g1_lafan1_dance1_subject2",
    url=(
        "https://raw.githubusercontent.com/KraHsu/genelab-assets/main/"
        "unitree_g1/motions/dance1_subject2.npz"
    ),
    md5="844731ab25e33ccd67798d4e22067ff9",
    filename="dance1_subject2.npz",
)


def g1_lafan1_dance1_subject2() -> Path:
    """Return cached path to the LAFAN1 `dance1_subject2` G1 motion NPZ.

    Downloads from the GeneLab asset zoo on first use and caches under
    ``.cache/assets/g1_lafan1_dance1_subject2/<md5>/dance1_subject2.npz``.
    """
    return fetch_asset(_DANCE1_SUBJECT2)
