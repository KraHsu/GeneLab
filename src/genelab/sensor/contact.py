"""Per-link contact sensor with optional air-time / contact-time state machine.

The shape mirrors mjlab's ``ContactSensor`` — same ``found`` / ``force`` / ``current_air_time`` /
``last_air_time`` semantics so obs / reward terms transfer 1:1 — but the backend reads Genesis's
``robot.get_links_net_contact_force()`` (per-link external-contact aggregate) instead of MuJoCo
contact pairs. A contact is "found" when the force magnitude exceeds ``force_threshold`` (N).
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class ContactData:
    """Per-step contact snapshot. All tensors are shaped ``(num_envs, num_links)`` unless noted."""

    force: torch.Tensor  # (B, N, 3) world-frame net contact force per selected link
    force_norm: torch.Tensor  # (B, N) magnitude of ``force``
    found: torch.Tensor  # (B, N) bool: force_norm > force_threshold
    current_air_time: torch.Tensor  # (B, N) seconds since last lift-off; 0 while in contact
    last_air_time: torch.Tensor  # (B, N) duration of the most recently completed air phase
    current_contact_time: torch.Tensor  # (B, N) seconds since last landing; 0 while airborne
    last_contact_time: torch.Tensor  # (B, N) duration of the most recently completed contact
    first_contact: torch.Tensor  # (B, N) bool: this step transitioned from air → contact
    first_detached: torch.Tensor  # (B, N) bool: this step transitioned from contact → air


@dataclass
class ContactSensorCfg(SensorCfg):
    """Configuration for ``ContactSensor``.

    Provide either an explicit tuple of ``link_names`` or a regex ``link_names_expr`` (matched
    against ``env.link_names`` with ``re.search``). ``track_air_time`` toggles allocation of the
    air-time / contact-time state machine; turn it off to skip the per-step state update on
    sensors that only need ``force`` / ``found``.
    """

    link_names: tuple[str, ...] = ()
    link_names_expr: str | None = None
    force_threshold: float = 1.0
    track_air_time: bool = True

    def build(self) -> "ContactSensor":
        return ContactSensor(self)


@dataclass
class _AirTimeState:
    current_air_time: torch.Tensor
    last_air_time: torch.Tensor
    current_contact_time: torch.Tensor
    last_contact_time: torch.Tensor
    first_contact: torch.Tensor
    first_detached: torch.Tensor


def _resolve_link_indices(
    cfg: ContactSensorCfg, link_names: list[str]
) -> tuple[list[int], list[str]]:
    if cfg.link_names_expr is not None:
        pattern = re.compile(cfg.link_names_expr)
        matched = [(i, n) for i, n in enumerate(link_names) if pattern.search(n)]
    elif cfg.link_names:
        missing = [n for n in cfg.link_names if n not in link_names]
        if missing:
            raise ValueError(
                f"ContactSensorCfg(name={cfg.name!r}): link(s) {missing!r} not in env.link_names"
            )
        order = {n: i for i, n in enumerate(link_names)}
        matched = [(order[n], n) for n in cfg.link_names]
    else:
        raise ValueError(
            f"ContactSensorCfg(name={cfg.name!r}) requires either link_names or link_names_expr"
        )
    if not matched:
        raise ValueError(
            f"ContactSensorCfg(name={cfg.name!r}): "
            f"link_names_expr={cfg.link_names_expr!r} matched nothing"
        )
    return [i for i, _ in matched], [n for _, n in matched]


class ContactSensor(Sensor[ContactData]):
    def __init__(self, cfg: ContactSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx_tensor: torch.Tensor | None = None
        self._resolved_link_names: list[str] = []
        self._air_state: _AirTimeState | None = None

    @property
    def link_names(self) -> list[str]:
        return list(self._resolved_link_names)

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        indices, names = _resolve_link_indices(self._cfg_typed, env.link_names)
        self._resolved_link_names = names
        self._link_idx_tensor = torch.tensor(indices, dtype=torch.long, device=env.device)
        if self._cfg_typed.track_air_time:
            shape = (env.num_envs, len(indices))
            self._air_state = _AirTimeState(
                current_air_time=torch.zeros(*shape, device=env.device),
                last_air_time=torch.zeros(*shape, device=env.device),
                current_contact_time=torch.zeros(*shape, device=env.device),
                last_contact_time=torch.zeros(*shape, device=env.device),
                first_contact=torch.zeros(*shape, dtype=torch.bool, device=env.device),
                first_detached=torch.zeros(*shape, dtype=torch.bool, device=env.device),
            )

    def update(self, dt: float) -> None:
        super().update(dt)
        if self._air_state is None or self._env is None:
            return
        force = self._read_link_forces()
        in_contact = force.norm(dim=-1) > self._cfg_typed.force_threshold
        state = self._air_state
        # Capture transitions before mutating the timers (the timer values matter pre-tick).
        # Persisting them as state lets consumers ask "did a foot just land this step?" via
        # ``data.first_contact`` — used by ``soft_landing`` / ``feet_swing_height`` rewards.
        first_contact = in_contact & (state.current_air_time > 0)
        first_detached = (~in_contact) & (state.current_contact_time > 0)
        state.first_contact = first_contact
        state.first_detached = first_detached
        state.last_air_time = torch.where(
            first_contact, state.current_air_time + dt, state.last_air_time
        )
        state.last_contact_time = torch.where(
            first_detached, state.current_contact_time + dt, state.last_contact_time
        )
        # Tick all timers, then mask-zero whichever phase ended. Original ``torch.where`` with
        # ``torch.zeros_like`` allocated two (B, N) tensors per step; this in-place form is
        # zero-allocation and removes the dominant per-step churn from the contact sensor.
        state.current_air_time += dt
        state.current_air_time.masked_fill_(in_contact, 0.0)
        state.current_contact_time += dt
        state.current_contact_time.masked_fill_(~in_contact, 0.0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if self._air_state is None or self._env is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        for buf in (
            self._air_state.current_air_time,
            self._air_state.last_air_time,
            self._air_state.current_contact_time,
            self._air_state.last_contact_time,
        ):
            buf[env_ids] = 0.0
        self._air_state.first_contact[env_ids] = False
        self._air_state.first_detached[env_ids] = False

    def _compute_data(self) -> ContactData:
        force = self._read_link_forces()
        force_norm = force.norm(dim=-1)
        found = force_norm > self._cfg_typed.force_threshold
        if self._air_state is None:
            zeros = torch.zeros_like(force_norm)
            falses = torch.zeros_like(found)
            return ContactData(
                force=force,
                force_norm=force_norm,
                found=found,
                current_air_time=zeros,
                last_air_time=zeros,
                current_contact_time=zeros.clone(),
                last_contact_time=zeros.clone(),
                first_contact=falses,
                first_detached=falses.clone(),
            )
        s = self._air_state
        return ContactData(
            force=force,
            force_norm=force_norm,
            found=found,
            current_air_time=s.current_air_time,
            last_air_time=s.last_air_time,
            current_contact_time=s.current_contact_time,
            last_contact_time=s.last_contact_time,
            first_contact=s.first_contact,
            first_detached=s.first_detached,
        )

    def _read_link_forces(self) -> torch.Tensor:
        """Return per-selected-link external contact force ``(num_envs, num_links, 3)``."""
        assert self._env is not None and self._link_idx_tensor is not None
        robot = self._env.robot
        getter = getattr(robot, "get_links_net_contact_force", None)
        if getter is None:
            # Genesis fake / not built — return zeros so the abstraction stays usable in tests.
            return torch.zeros(
                self._env.num_envs,
                self._link_idx_tensor.numel(),
                3,
                device=self._env.device,
            )
        try:
            forces = getter()
        except Exception:
            return torch.zeros(
                self._env.num_envs,
                self._link_idx_tensor.numel(),
                3,
                device=self._env.device,
            )
        forces = forces.to(self._env.device)
        if forces.dim() == 2:
            forces = forces.unsqueeze(0).expand(self._env.num_envs, -1, -1)
        return forces.index_select(-2, self._link_idx_tensor)
