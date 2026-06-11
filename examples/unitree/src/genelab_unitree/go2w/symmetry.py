"""Mirror-symmetry augmentation for the Go2-W velocity task (rsl_rl ``Symmetry``).

The robot and the velocity-tracking task are mirror-symmetric about the x-z plane, so a
y-flip maps every (+vy, +wz) experience onto a valid (−vy, −wz) experience. Doubling each
PPO mini-batch with mirrored samples makes both lateral directions learn in lock-step —
the structural fix for the one-sided gait collapse seen across stage-2 fine-tunes (one
lateral direction perfect, the mirror direction falling 64/64, flipping side between runs).

Under the y-flip:

* linear quantities negate y (lin vel, gravity); angular quantities negate x and z;
* left/right joints swap within each articulation 4-group ``[FL, FR, RL, RR]``;
* abduction (hip, x-axis) angles negate; thigh / calf / wheel (y-axis) keep sign.

The index/sign maps below are tied to the go2w env's frame layout (declared term order ×
``_OBS_HISTORY`` frame-major stacking) and the action layout (12 leg position targets +
4 wheel velocities). ``mirror_go2w_obs_actions`` asserts the widths so a layout change
fails loudly instead of silently training on garbage mirrors.
"""

from typing import Any

import torch

# Per-frame policy obs layout: ang_vel(3) gravity(3) joint_pos(16) joint_vel(16)
# actions(16) commands(3). Critic appends base_lin_vel(3).
POLICY_FRAME_DIM = 57
CRITIC_FRAME_DIM = 60
HISTORY = 5
ACTION_DIM = 16

# Articulation 4-group [FL, FR, RL, RR] -> [FR, FL, RR, RL].
_SWAP4 = [1, 0, 3, 2]


def _joint16() -> tuple[list[int], list[float]]:
    """(perm, sign) for a 16-wide joint vector: hip(−)·thigh·calf·wheel groups of 4."""
    perm: list[int] = []
    sign: list[float] = []
    for group, negate in ((0, True), (1, False), (2, False), (3, False)):
        base = group * 4
        perm.extend(base + i for i in _SWAP4)
        sign.extend([-1.0 if negate else 1.0] * 4)
    return perm, sign


def _action16() -> tuple[list[int], list[float]]:
    """(perm, sign) for the 16-wide action: leg targets hip(−)·thigh·calf + wheels."""
    return _joint16()  # same group structure: hip/thigh/calf/wheel × 4


def _frame_maps(frame_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (perm, sign) tensors for one observation frame of ``frame_dim`` columns."""
    j_perm, j_sign = _joint16()
    a_perm, a_sign = _action16()
    perm: list[int] = []
    sign: list[float] = []

    def block(offset: int, p: list[int], s: list[float]) -> None:
        perm.extend(offset + i for i in p)
        sign.extend(s)

    block(0, [0, 1, 2], [-1.0, 1.0, -1.0])  # base_ang_vel: wx, wz negate
    block(3, [0, 1, 2], [1.0, -1.0, 1.0])  # projected_gravity: gy negates
    block(6, j_perm, j_sign)  # joint_pos
    block(22, j_perm, j_sign)  # joint_vel
    block(38, a_perm, a_sign)  # last action
    block(54, [0, 1, 2], [1.0, -1.0, -1.0])  # commands: vy, wz negate
    if frame_dim == CRITIC_FRAME_DIM:
        block(57, [0, 1, 2], [1.0, -1.0, 1.0])  # privileged base_lin_vel: vy negates
    return torch.tensor(perm, dtype=torch.long), torch.tensor(sign)


def _stacked_maps(frame_dim: int, history: int) -> tuple[torch.Tensor, torch.Tensor]:
    perm, sign = _frame_maps(frame_dim)
    offsets = torch.arange(history).repeat_interleave(frame_dim) * frame_dim
    return perm.repeat(history) + offsets, sign.repeat(history)


_MAPS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
    POLICY_FRAME_DIM * HISTORY: _stacked_maps(POLICY_FRAME_DIM, HISTORY),
    CRITIC_FRAME_DIM * HISTORY: _stacked_maps(CRITIC_FRAME_DIM, HISTORY),
}
_ACTION_MAPS: tuple[torch.Tensor, torch.Tensor] = (
    torch.tensor(_action16()[0], dtype=torch.long),
    torch.tensor(_action16()[1]),
)


def _mirror(t: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return t.index_select(-1, perm.to(t.device)) * sign.to(t.device)


def mirror_go2w_obs_actions(
    env: Any, obs: Any = None, actions: torch.Tensor | None = None
) -> tuple[Any, torch.Tensor | None]:
    """rsl_rl ``Symmetry`` data-augmentation hook: return ``[orig; mirrored]`` batches.

    ``obs`` is the rollout observation container (a TensorDict / mapping of group name to
    tensor, here ``policy`` (285) and ``critic`` (300)); ``actions`` is the ``(B, 16)``
    action tensor. Either may be ``None`` (the extension calls with one side only).
    """
    del env
    obs_out: Any = obs
    if obs is not None:
        mirrored = {}
        for key in obs.keys():
            t = obs[key]
            width = int(t.shape[-1])
            if width not in _MAPS:
                raise ValueError(
                    f"go2w mirror: obs group {key!r} width {width} does not match the "
                    f"expected stacked layouts {sorted(_MAPS)} — env layout changed?"
                )
            perm, sign = _MAPS[width]
            mirrored[key] = _mirror(t, perm, sign)
        if hasattr(obs, "batch_size"):  # TensorDict path: keep container type + metadata
            obs_out = torch.cat([obs, obs.clone().update(mirrored)], dim=0)
        else:
            obs_out = {k: torch.cat([obs[k], mirrored[k]], dim=0) for k in mirrored}
    act_out = actions
    if actions is not None:
        if int(actions.shape[-1]) != ACTION_DIM:
            raise ValueError(f"go2w mirror: action width {int(actions.shape[-1])} != {ACTION_DIM}")
        perm, sign = _ACTION_MAPS
        act_out = torch.cat([actions, _mirror(actions, perm, sign)], dim=0)
    return obs_out, act_out
