"""Self-collision force sensor.

Wraps Genesis's per-pair contact list (``robot.get_contacts(with_entity=robot)``)
to report a single per-env self-contact force magnitude — the sum of
``|force|`` across every contact pair where both geoms live inside the same
entity. Counterpart to the existing :class:`~genelab.sensor.contact.ContactSensor`,
which only sees per-link *net* contact force from all sources (and so cannot
distinguish foot-vs-ground from limb-vs-limb).

Use case: the ``self_collision_cost`` reward (mjlab parity) needs a robust
"is the robot kicking / hitting itself" signal — typically with a short
``history_length`` window so transient sub-step impacts don't slip between
policy steps. mjlab's reference G1 config uses ``history_length=4`` on a
``num_slots=1, reduce="none"`` setup; this sensor matches that shape (one
scalar per env, optional rolling window).

The simpler per-env scalar API (vs mjlab's per-slot tensor) is intentional —
the only consumer is a counter-of-hits reward, and the slot-level breakdown
mjlab carries is unused there.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class SelfContactData:
    """Per-step self-contact snapshot. All tensors shaped ``(B,)`` unless noted."""

    force: torch.Tensor  # (B,) sum of |force| across all self-contact pairs this step
    found: torch.Tensor  # (B,) bool: force > force_threshold
    # ``(B, H)`` rolling window of recent per-env force sums in write-pointer order.
    # ``None`` when ``history_length=0``. Order is irrelevant for typical consumers
    # (``any`` / ``sum`` aggregations).
    force_history: torch.Tensor | None


@dataclass
class SelfContactSensorCfg(SensorCfg):
    """Configuration for :class:`SelfContactSensor`.

    Unlike :class:`~genelab.sensor.contact.ContactSensorCfg` this sensor takes no
    ``link_names`` — Genesis's pair-filtered contact list already restricts the
    output to within-entity contacts, and the reward consuming it doesn't need
    per-link breakdown. If a future use case wants per-link self-contact force,
    a follow-up can extend the aggregator with a ``link_names`` filter.
    """

    force_threshold: float = 1.0
    history_length: int = 0

    def build(self) -> "SelfContactSensor":
        return SelfContactSensor(self)


class SelfContactSensor(Sensor[SelfContactData]):
    def __init__(self, cfg: SelfContactSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._latest_force: torch.Tensor | None = None
        self._force_history: torch.Tensor | None = None
        self._history_head: int = 0

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        self._latest_force = torch.zeros(env.num_envs, device=env.device)
        if self._cfg_typed.history_length > 0:
            self._force_history = torch.zeros(
                env.num_envs, self._cfg_typed.history_length, device=env.device
            )
            self._history_head = 0

    def update(self, dt: float) -> None:
        super().update(dt)
        if self._env is None:
            return
        # Compute once per step and reuse in ``_compute_data`` — Genesis's
        # ``get_contacts`` walks the full pair list, which is expensive enough
        # to be worth the cache.
        self._latest_force = self._compute_per_env_force()
        if self._force_history is not None:
            history_length = self._cfg_typed.history_length
            self._force_history[:, self._history_head] = self._latest_force
            self._history_head = (self._history_head + 1) % history_length

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if self._env is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        if self._latest_force is not None:
            self._latest_force[env_ids] = 0.0
        if self._force_history is not None:
            # Clear the rolling window so prior-episode hits don't leak into the new episode.
            # The global head pointer is unchanged; stale slots get overwritten next step.
            self._force_history[env_ids] = 0.0

    def _compute_per_env_force(self) -> torch.Tensor:
        """Sum of per-pair force magnitudes for valid self-contact pairs.

        Falls back to zeros when Genesis isn't built (tests using fake robots) or
        when ``get_contacts`` raises for any reason — this keeps the sensor usable
        in the same fake-env scaffolding the rest of the suite uses.
        """
        assert self._env is not None
        robot = self._env.robot
        getter = getattr(robot, "get_contacts", None)
        if getter is None:
            return torch.zeros(self._env.num_envs, device=self._env.device)
        try:
            contacts = getter(with_entity=robot)
        except Exception:
            return torch.zeros(self._env.num_envs, device=self._env.device)
        force_a = contacts.get("force_a") if isinstance(contacts, dict) else None
        valid = contacts.get("valid_mask") if isinstance(contacts, dict) else None
        if force_a is None or valid is None:
            return torch.zeros(self._env.num_envs, device=self._env.device)
        force_a = force_a.to(self._env.device)
        valid = valid.to(self._env.device)
        if force_a.dim() == 2:
            # Single-env Genesis path returns pre-filtered (n_contacts, 3); add the batch
            # dim back so the downstream shape stays consistent. ``valid_mask`` is absent
            # in that mode, so we treat all rows as valid.
            force_a = force_a.unsqueeze(0).expand(self._env.num_envs, -1, -1)
            valid = torch.ones_like(force_a[..., 0], dtype=torch.bool)
        force_mag = force_a.norm(dim=-1)  # (B, n_contacts)
        return (force_mag * valid.float()).sum(dim=-1)

    def _compute_data(self) -> SelfContactData:
        assert self._latest_force is not None
        force = self._latest_force
        found = force > self._cfg_typed.force_threshold
        return SelfContactData(
            force=force,
            found=found,
            force_history=self._force_history,
        )
