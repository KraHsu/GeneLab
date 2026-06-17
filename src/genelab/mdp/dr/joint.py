"""Joint-level domain randomization (currently: encoder-bias sim2real DR)."""

from typing import TYPE_CHECKING

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import asset_articulation, asset_handle, asset_state
from genelab.mdp.dr._common import normalise_env_ids, resolve_joint_indices

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def randomize_joint_stiffness_damping(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    stiffness_range: tuple[float, float] = (0.8, 1.2),
    damping_range: tuple[float, float] = (0.8, 1.2),
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Per-env multiplicative DR on each actuator group's PD gains (kp / kv).

    For every actuator the kp and kv multipliers are sampled per-env, per-joint:

    * **force-channel** actuators (IdealPD / DCMotor / MlpResidual) receive a per-env
      gain *scale* applied inside their Python ``compute`` (via ``set_gain_scale``);
    * **implicit-PD** actuators have their sim-side kp/kv rewritten to
      ``nominal × multiplier`` per env through Genesis ``set_dofs_kp`` / ``set_dofs_kv``.

    Multipliers default to ±20% — the standard sim2real PD-gain calibration sweep.
    Genesis calls are guarded so the function stays usable on the fake-env test
    scaffolding.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    handle = asset_handle(env, asset_cfg)
    set_kp = getattr(handle, "set_dofs_kp", None)
    set_kv = getattr(handle, "set_dofs_kv", None)
    for actuator in asset_articulation(env, asset_cfg).actuators.values():
        n_joints = actuator.num_joints
        if n_joints == 0:
            continue
        kp_mult = torch.empty(n, n_joints, device=env.device).uniform_(*stiffness_range)
        kv_mult = torch.empty(n, n_joints, device=env.device).uniform_(*damping_range)
        if actuator.channel == "force":
            actuator.set_gain_scale(env_ids, kp_mult, kv_mult)
            continue
        kp_vals = actuator.stiffness.unsqueeze(0) * kp_mult
        kv_vals = actuator.damping.unsqueeze(0) * kv_mult
        if set_kp is not None:
            try:
                set_kp(kp_vals, dofs_idx_local=actuator.dof_ids, envs_idx=env_ids)
            except Exception:
                pass
        if set_kv is not None:
            try:
                set_kv(kv_vals, dofs_idx_local=actuator.dof_ids, envs_idx=env_ids)
            except Exception:
                pass


def dof_frictionloss(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    friction: float = 0.01,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Set a GLOBAL joint dry-friction (frictionloss, Nm) baseline on the actuated joints.

    The hand MJCF declares zero joint frictionloss, so a Genesis-trained policy never
    learns to overcome the *real* hand's static friction — it under-drives the fingers and
    reorients too slowly on hardware (the cube is held but the goal times out). Adding a
    realistic stiction baseline forces the policy to drive through it, which transfers.

    NOTE: Genesis frictionloss is a *non-batched* dof property (shared across all envs), so
    this is a single global value, not per-env DR (unlike kp/kv). Use ``mode="startup"``.
    Written via ``set_dofs_frictionloss``; guarded for the fake-env test scaffolding.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    handle = asset_handle(env, asset_cfg)
    setter = getattr(handle, "set_dofs_frictionloss", None) or getattr(
        handle, "set_dofs_friction", None
    )
    if setter is None:
        return
    for actuator in asset_articulation(env, asset_cfg).actuators.values():
        n_joints = actuator.num_joints
        if n_joints == 0:
            continue
        vals = torch.full((n_joints,), float(friction), device=env.device)  # 1D = global
        try:
            setter(vals, dofs_idx_local=actuator.dof_ids)
        except TypeError:
            try:
                setter(vals, actuator.dof_ids)
            except Exception:
                pass
        except Exception:
            pass


def encoder_bias(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bias_range: tuple[float, float] = (-0.015, 0.015),
) -> None:
    """Sample a per-env, per-joint encoder bias and write it into ``robot_state``.

    Models the small constant offset between the policy's perceived joint angle
    (the encoder reading) and the physical reality.
    :class:`~genelab.mdp.actions.joint_position.JointPositionAction` subtracts
    the bias from its PD target, so the joint settles at ``-bias`` relative to
    the policy's nominal command. The observation
    (:func:`genelab.mdp.observations.joint_pos_rel`) returns the raw
    ``joint_pos − default`` and therefore exposes that offset directly — the
    policy must learn to compensate, which is the sim2real-hardening behaviour
    Isaac Lab gets from the same DR.

    Isaac Lab uses ``(-0.015, 0.015)`` rad (≈0.86°) for the G1 velocity task.
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
    encoder_bias_buf = asset_state(env, asset_cfg).encoder_bias
    if asset_cfg.joint_ids is None:
        # Optimisation: hitting every joint, so a flat row-slice avoids the 2-d
        # advanced indexing overhead. Behavior is identical to the else branch.
        encoder_bias_buf[env_ids] = samples
    else:
        # 2-d advanced indexing: outer product of env rows × joint columns. The
        # ``env_ids[:, None]`` / ``joint_ids[None, :]`` broadcast produces the
        # correct ``(n_envs, n_joints)`` write target.
        encoder_bias_buf[env_ids[:, None], joint_ids_tensor[None, :]] = samples
