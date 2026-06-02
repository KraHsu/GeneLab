"""Hand-object contact sensor built on Genesis ``get_contacts``.

Genesis exposes the full per-pair contact list (``link_a``/``link_b``/``force``) via
``RigidEntity.get_contacts(with_entity=...)``. This sensor restricts to robot-vs-cube
pairs and aggregates them into the signals the reorient rewards need:

* ``tip_found`` / ``tip_force`` — per-fingertip (link4) contact with the cube,
* ``palm_found`` — palm in contact with the cube,
* ``distal_found`` — any distal phalanx (link3 / link4) in contact with the cube.

Finger self-collision uses the stock :class:`~genelab.sensor.SelfContactSensor`.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor._entity import entity_articulation, entity_handle
from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_TIP_RE = re.compile(r"right_finger[1-5]_link4$")
_DISTAL_RE = re.compile(r"right_finger[1-5]_link[34]$")
_PALM_RE = re.compile(r".*_palm_link$")


@dataclass
class HandObjectContactData:
    """Per-step hand-cube contact snapshot."""

    tip_found: torch.Tensor  # (B, 5) bool per fingertip
    tip_force: torch.Tensor  # (B, 5) contact-force magnitude per fingertip
    palm_found: torch.Tensor  # (B,) bool
    distal_found: torch.Tensor  # (B,) bool


@dataclass
class HandObjectContactSensorCfg(SensorCfg):
    object_name: str = "object"
    force_threshold: float = 1.0

    def build(self) -> "HandObjectContactSensor":
        return HandObjectContactSensor(self)


class HandObjectContactSensor(Sensor[HandObjectContactData]):
    def __init__(self, cfg: HandObjectContactSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._tip_ids: torch.Tensor | None = None
        self._distal_ids: torch.Tensor | None = None
        self._palm_id: int = -1
        self._data: HandObjectContactData | None = None

    def bind(self, env: "EnvContext") -> None:
        super().bind(env)
        names = entity_articulation(env, self._cfg.entity_name).link_names
        self._tip_ids = torch.tensor(
            [i for i, n in enumerate(names) if _TIP_RE.search(n)], device=env.device
        )
        self._distal_ids = torch.tensor(
            [i for i, n in enumerate(names) if _DISTAL_RE.search(n)], device=env.device
        )
        self._palm_id = next(i for i, n in enumerate(names) if _PALM_RE.search(n))

    def _zeros(self) -> HandObjectContactData:
        assert self._env is not None and self._tip_ids is not None
        n, dev = self._env.num_envs, self._env.device
        return HandObjectContactData(
            tip_found=torch.zeros(n, self._tip_ids.numel(), dtype=torch.bool, device=dev),
            tip_force=torch.zeros(n, self._tip_ids.numel(), device=dev),
            palm_found=torch.zeros(n, dtype=torch.bool, device=dev),
            distal_found=torch.zeros(n, dtype=torch.bool, device=dev),
        )

    def update(self, dt: float) -> None:
        super().update(dt)
        self._data = self._compute_contacts()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self._data = None

    def _compute_contacts(self) -> HandObjectContactData:
        assert self._env is not None and self._tip_ids is not None and self._distal_ids is not None
        robot = entity_handle(self._env, self._cfg.entity_name)
        cube = self._env.scene[self._cfg_typed.object_name].gs_handle  # type: ignore[index]
        getter = getattr(robot, "get_contacts", None)
        if getter is None or getattr(robot, "link_start", None) is None:
            return self._zeros()
        try:
            c = getter(with_entity=cube)
        except Exception:
            return self._zeros()
        if not isinstance(c, dict) or "link_a" not in c:
            return self._zeros()
        dev = self._env.device
        la = c["link_a"].to(dev)
        lb = c["link_b"].to(dev)
        force = c["force_a"].to(dev)
        valid = c.get("valid_mask")
        if la.dim() == 1:  # single-env: (nc,) — add batch dim, all valid
            la, lb = la.unsqueeze(0), lb.unsqueeze(0)
            force = force.unsqueeze(0)
            valid = torch.ones_like(la, dtype=torch.bool)
        else:
            valid = torch.ones_like(la, dtype=torch.bool) if valid is None else valid.to(dev)
        if la.shape[1] == 0:  # no contact pairs this step — nothing to aggregate
            return self._zeros()
        ls, le = int(robot.link_start), int(robot.link_end)
        a_is_robot = (la >= ls) & (la < le)
        robot_link = torch.where(a_is_robot, la, lb) - ls  # (B, nc) local link
        fmag = force.norm(dim=-1)  # (B, nc)
        hit = valid & (fmag > self._cfg_typed.force_threshold)  # (B, nc)

        tip_match = robot_link.unsqueeze(-1) == self._tip_ids.view(1, 1, -1)  # (B, nc, 5)
        tip_hit = hit.unsqueeze(-1) & tip_match
        tip_found = tip_hit.any(dim=1)
        tip_force = torch.where(
            tip_hit, fmag.unsqueeze(-1), torch.zeros_like(tip_hit, dtype=fmag.dtype)
        ).amax(dim=1)
        palm_found = (hit & (robot_link == self._palm_id)).any(dim=1)
        distal_match = (robot_link.unsqueeze(-1) == self._distal_ids.view(1, 1, -1)).any(dim=-1)
        distal_found = (hit & distal_match).any(dim=1)
        return HandObjectContactData(
            tip_found=tip_found,
            tip_force=tip_force,
            palm_found=palm_found,
            distal_found=distal_found,
        )

    def _compute_data(self) -> HandObjectContactData:
        return self._data if self._data is not None else self._zeros()
