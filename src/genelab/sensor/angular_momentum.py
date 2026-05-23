"""Root angular momentum sensor (orbital approximation).

Computes ``L = Σᵢ mᵢ · (rᵢ − r_root) × vᵢ`` in the world frame, where the sum
runs over every link in the articulation. This is the **orbital** half of the
true angular momentum; the per-link spin term ``Σᵢ Iᵢ_world · ωᵢ`` is omitted
because Genesis does not expose per-link inertia tensors through a stable
public API and reading internals (``rigid_link._variant_inertial``) would
couple the sensor to engine version. For the locomotion penalty
``mdp.angular_momentum_penalty`` (mjlab parity, weight −0.02) the orbital
term dominates — a humanoid's "spinning" failure mode is dominated by limbs
swinging at long lever arms, not by individual links spinning about their
own axes. Document the approximation in callers expecting tight physical
fidelity.

Output: ``(B, 3)`` world-frame angular momentum vector per env.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_handle, entity_state
from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class RootAngularMomentumSensorCfg(SensorCfg):
    """Configuration for :class:`RootAngularMomentumSensor`.

    Takes no parameters — every articulated link contributes to the orbital sum, and
    the reference frame is fixed at the root link's world position. A future
    extension could add link-name filtering (e.g. compute angular momentum of upper
    body only) if a use case appears.
    """

    def build(self) -> "RootAngularMomentumSensor":
        return RootAngularMomentumSensor(self)


class RootAngularMomentumSensor(Sensor[torch.Tensor]):
    """Per-env world-frame angular momentum about the root link position."""

    def __init__(self, cfg: RootAngularMomentumSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        # Per-link masses cached at bind. ``None`` until ``bind`` runs; once bound
        # the tensor is shaped (B, num_links) so the broadcast in ``_compute_data``
        # is allocation-free per step.
        self._link_masses: torch.Tensor | None = None

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        num_links = len(entity_articulation(env, self._cfg.entity_name).link_names)
        # Masses are static (unless P4's body-mass DR mutates them; we'll need a refresh
        # hook then). Fetch once and broadcast across envs. ``get_links_inertial_mass`` may
        # return (num_links,) or (n_envs, n_links) depending on Genesis version — normalise.
        masses_per_env_links = torch.zeros(env.num_envs, num_links, device=env.device)
        getter = getattr(entity_handle(env, self._cfg.entity_name), "get_links_inertial_mass", None)
        if getter is not None:
            try:
                m = getter()
            except Exception:
                m = None
            if m is not None:
                m = m.to(env.device)
                if m.dim() == 1:
                    masses_per_env_links = m.unsqueeze(0).expand(env.num_envs, -1).contiguous()
                elif m.dim() == 2:
                    masses_per_env_links = m.contiguous()
        self._link_masses = masses_per_env_links

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None and self._link_masses is not None
        state = entity_state(self._env, self._cfg.entity_name)
        link_pos = state.link_pos  # (B, num_links, 3)
        link_vel = state.link_lin_vel_w  # (B, num_links, 3)
        root_pos = state.root_pos  # (B, 3)
        # Position relative to root, then orbital cross-product. Sum mass-weighted contributions.
        r_rel = link_pos - root_pos.unsqueeze(1)
        cross = torch.linalg.cross(r_rel, link_vel, dim=-1)  # (B, num_links, 3)
        return (self._link_masses.unsqueeze(-1) * cross).sum(dim=1)
