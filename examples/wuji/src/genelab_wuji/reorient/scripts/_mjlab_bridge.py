"""Adapter for running the GeneLab-trained reorient policy inside wuji-mjlab's own env.

wuji-mjlab is the MuJoCo/mjlab reference this task was ported from. Its observation is *not*
identical to GeneLab's: mjlab feeds limit-normalized joint angles + joint-position target
error + a tag-frame goal error, while GeneLab feeds joint-position-relative + joint velocity +
a world-frame goal error — both as a 3-step history. So the policy cannot be dropped into
mjlab's native obs pipeline; these helpers rebuild the GeneLab actor observation from raw
state (with the Genesis joint-major <-> mjlab finger-major remap and the 3-step history), so
the policy sees exactly what it trained on while the *simulation* stays 100% wuji-mjlab.

Requires ``wuji_mjlab`` importable; run inside its environment (e.g. its pixi env) with
GeneLab on ``PYTHONPATH``.
"""

import torch
from torch import nn

from genelab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)
from genelab_wuji.reorient.constants import (
    REORIENT_HISTORY_LEN,
    REORIENT_JOINT_POS,
    REORIENT_ROBOT_ROOT_POS,
    REORIENT_ROBOT_ROOT_ROT,
    TAG_IN_PALM_POS,
    TAG_IN_PALM_QUAT_WXYZ,
)
from genelab_wuji.reorient.mdp._math import matrix_to_rotation_6d

# GeneLab (Genesis) enumerates joints joint-major; mjlab is finger-major.
POLICY_JOINTS = tuple(f"right_finger{f}_joint{j}" for j in range(1, 5) for f in range(1, 6))
MJLAB_JOINTS = tuple(f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))


def load_policy(checkpoint: str, device: str = "cpu"):
    """Rebuild the actor (EmpiricalNormalization + MLP) from an rsl_rl checkpoint.

    The obs dimension is read from the checkpoint, so this works for both the 1-step (69) and
    3-step (207) observation variants. The TorchScript export drops the normalizer, hence
    loading from ``actor_state_dict`` directly.
    """
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    in_dim = int(sd["mlp.0.weight"].shape[1])
    mean = sd["obs_normalizer._mean"].float().to(device)
    std = sd["obs_normalizer._std"].float().to(device)
    mlp = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 20),
    )
    mlp.load_state_dict({k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    mlp.to(device)

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        return mlp((obs - mean) / std)

    policy.obs_dim = in_dim  # type: ignore[attr-defined]
    return policy


def tag_frame(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame wrist tag pose (constant, fixed-base hand) — GeneLab's training frame."""
    root_pos = torch.tensor(REORIENT_ROBOT_ROOT_POS, device=device)
    root_quat = torch.tensor(REORIENT_ROBOT_ROOT_ROT, device=device)
    tag_quat = quat_mul(
        root_quat.unsqueeze(0), torch.tensor(TAG_IN_PALM_QUAT_WXYZ, device=device).unsqueeze(0)
    )[0]
    tag_pos = (
        root_pos
        + quat_apply(
            root_quat.unsqueeze(0), torch.tensor(TAG_IN_PALM_POS, device=device).unsqueeze(0)
        )[0]
    )
    return tag_pos, tag_quat


def goal_err_6d(cube_quat: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
    """GeneLab world-frame cube-to-goal 6D rotation error (matches the training obs term)."""
    err = quat_mul(cube_quat.unsqueeze(0), quat_conjugate(goal_quat.unsqueeze(0)))
    return matrix_to_rotation_6d(matrix_from_quat(err))[0]


def default_policy_joint_pos(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([REORIENT_JOINT_POS[n] for n in POLICY_JOINTS], device=device)


class ObsHistory:
    """3-step, term-major observation history (oldest->newest), backfilled on reset.

    Mirrors the training-time history in ``observations._with_history`` so the policy receives
    the same 207-dim observation in the bridge as in Genesis training.
    """

    def __init__(self, history_length: int = REORIENT_HISTORY_LEN) -> None:
        self.h = history_length
        self._buf: dict[str, list[torch.Tensor]] = {}

    def reset(self) -> None:
        self._buf = {}

    def _push(self, key: str, value: torch.Tensor) -> torch.Tensor:
        frames = self._buf.get(key)
        if frames is None:
            frames = [value.clone() for _ in range(self.h)]
        else:
            frames = frames[1:] + [value.clone()]
        self._buf[key] = frames
        return torch.cat(frames)

    def build(
        self,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        cube_pos_in_tag: torch.Tensor,
        goal_rot_err_6d: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self._push("joint_pos", joint_pos_rel),
                self._push("joint_vel", joint_vel),
                self._push("cube_pos_in_tag", cube_pos_in_tag),
                self._push("goal_rot_err_6d", goal_rot_err_6d),
                self._push("last_action", last_action),
            ]
        )
