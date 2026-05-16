"""Joint-level domain randomization (currently: encoder-bias sim2real DR)."""

from typing import TYPE_CHECKING

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.dr._common import resolve_joint_indices

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def encoder_bias(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bias_range: tuple[float, float] = (-0.015, 0.015),
) -> None:
    """Sample a per-env, per-joint encoder bias and write it into ``robot_state``.

    Models the small constant offset between the policy's perceived joint angle
    (the encoder reading) and the physical reality. Once populated, the bias is
    read on both sides of the loop:

    * :func:`genelab.mdp.observations.joint_pos_rel` adds it to the observation
      so the policy sees a biased angle.
    * :class:`~genelab.mdp.actions.joint_position.JointPositionAction` subtracts
      it from the PD target so the joint settles ``bias`` away from where the
      policy thinks it commanded.

    Net effect: the policy's reference frame is silently shifted per joint per
    env; mjlab uses this with ``(-0.015, 0.015)`` rad (≈0.86°) to harden the G1
    policy against zero-point miscalibration.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    n_envs = int(env_ids.numel())
    joint_indices = resolve_joint_indices(env, asset_cfg)
    joint_ids_tensor = torch.tensor(joint_indices, dtype=torch.long, device=env.device)
    n_joints = int(joint_ids_tensor.numel())

    samples = torch.empty(n_envs, n_joints, device=env.device).uniform_(*bias_range)
    if asset_cfg.joint_ids is None:
        # Optimisation: hitting every joint, so a flat row-slice avoids the 2-d
        # advanced indexing overhead. Behavior is identical to the else branch.
        env.robot_state.encoder_bias[env_ids] = samples
    else:
        # 2-d advanced indexing: outer product of env rows × joint columns. The
        # ``env_ids[:, None]`` / ``joint_ids[None, :]`` broadcast produces the
        # correct ``(n_envs, n_joints)`` write target.
        env.robot_state.encoder_bias[env_ids[:, None], joint_ids_tensor[None, :]] = samples
