"""Actuator-level domain randomization (ROADMAP M2.1): per-env torque deadzone."""

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import asset_articulation
from genelab.mdp.dr._common import normalise_env_ids

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def randomize_actuator_deadzone(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    deadzone_range: tuple[float, float] = (0.0, 0.5),
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Per-env, per-joint actuator torque-deadzone half-width (N·m).

    Each force-channel actuator zeroes its computed effort where ``|τ| <`` the
    sampled deadzone (applied in the articulation control loop via
    :meth:`ActuatorBase.apply_deadzone`). Models actuator stiction / driver
    backlash — a small commanded torque produces no motion until it overcomes the
    deadband. Implicit-PD actuators (sim-side torque) are unaffected.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    for actuator in asset_articulation(env, asset_cfg).actuators.values():
        n_joints = actuator.num_joints
        if n_joints == 0:
            continue
        deadzone = torch.empty(n, n_joints, device=env.device).uniform_(*deadzone_range)
        actuator.set_deadzone(env_ids, deadzone)
