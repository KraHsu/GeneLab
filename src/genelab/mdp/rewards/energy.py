"""Energy-budget reward primitives.

Wraps Genesis 1.0's per-entity energy accessors —
``Articulation.gs_handle.get_kinetic_energy()`` /
``get_potential_energy()`` / ``get_total_energy()`` — into ``(num_envs,)``
reward terms. Useful as soft regularizers in sim-to-real curricula and
as hard-constraints on motion smoothness.

Each function takes the same ``asset_cfg`` selector pattern as the rest of
``genelab.mdp.rewards``: ``None`` resolves to the primary ``"robot"`` entity,
a populated :class:`SceneEntityCfg` routes to its named asset.

Sign conventions:

* :func:`kinetic_energy_l2` is non-negative (squared). Pair with a negative
  weight to penalize high-energy motion (motion-smoothness regularizer).
* :func:`potential_energy` is signed (Genesis returns negative values for
  centres-of-mass at world-frame height < 0); pair with a positive or
  negative weight depending on whether the task wants the body raised or
  lowered. The default ``base_height_l2`` is usually a better fit for
  base-height tracking; this primitive is for tasks that want the *full
  articulation*'s gravitational potential.
* :func:`energy_budget` is non-negative (squared deviation from a target
  total). Pair with a negative weight.
"""

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import asset_handle as _asset_handle

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def kinetic_energy_l2(env: "EnvContext", asset_cfg: "SceneEntityCfg | None" = None) -> torch.Tensor:
    """``T²`` per env, where ``T`` is the articulation's kinetic energy.

    Reads ``gs_handle.get_kinetic_energy()`` (Genesis 1.0) and squares the
    per-env scalar. Non-negative; pair with a negative weight to discourage
    high-energy motion.
    """
    ke = _asset_handle(env, asset_cfg).get_kinetic_energy()
    return ke * ke


def potential_energy(env: "EnvContext", asset_cfg: "SceneEntityCfg | None" = None) -> torch.Tensor:
    """Signed potential energy of the named articulation, in Joules.

    Reads ``gs_handle.get_potential_energy()`` directly. Sign follows
    Genesis: bodies at world-frame z < 0 carry negative potential, bodies
    above z = 0 carry positive potential. Pair with the appropriate signed
    weight for the task.
    """
    return _asset_handle(env, asset_cfg).get_potential_energy()


def energy_budget(
    env: "EnvContext",
    *,
    target_total: float = 0.0,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """``(total − target_total)²`` per env — squared deviation from a target total energy.

    Reads ``gs_handle.get_total_energy()`` (= kinetic + potential) and squares
    the deviation from ``target_total`` (in Joules). Non-negative; pair with
    a negative weight to keep the articulation's total mechanical energy near
    the target.
    """
    total = _asset_handle(env, asset_cfg).get_total_energy()
    delta = total - target_total
    return delta * delta
