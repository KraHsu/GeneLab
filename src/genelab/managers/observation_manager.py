"""Observation manager: per-group concatenated observation tensors."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext, NoiseCfg


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Observation term: optional noise / scale / clip applied to the raw tensor."""

    scale: float | None = None
    clip: tuple[float, float] | None = None
    noise: "NoiseCfg | None" = None


@dataclass
class ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    concatenate_terms: bool = True
    enable_corruption: bool = False
    # Frame stacking: feed the last ``history_length`` control-step frames concatenated
    # oldest->newest (frame-major). ``1`` (default) is the legacy single-frame behaviour.
    # A proprioception-only policy uses the stack to infer velocity / contact trends it
    # can't sense instantaneously — the main lever for robust sim2sim transfer.
    history_length: int = 1


class ObservationManager:
    """Computes a dict of group-name → flat tensor each control step."""

    def __init__(
        self,
        cfg: dict[str, ObservationGroupCfg],
        env: "EnvContext",
    ) -> None:
        self._env = env
        self.cfg: dict[str, ObservationGroupCfg] = deepcopy(cfg)
        self._group_terms: dict[str, list[tuple[str, ObservationTermCfg]]] = {}
        self._group_dims: dict[str, list[int]] = {}
        # Per-group frame-stacking buffer, shape (num_envs, history_length, frame_dim).
        self._history: dict[str, torch.Tensor] = {}
        for group_name, group_cfg in self.cfg.items():
            terms: list[tuple[str, ObservationTermCfg]] = []
            for term_name, term_cfg in group_cfg.terms.items():
                instantiate_class_term(term_cfg, env)
                terms.append((term_name, term_cfg))
            self._group_terms[group_name] = terms

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return {g: [n for n, _ in terms] for g, terms in self._group_terms.items()}

    def compute(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for group_name, terms in self._group_terms.items():
            group_cfg = self.cfg[group_name]
            tensors: list[torch.Tensor] = []
            for _, term_cfg in terms:
                value = term_cfg.func(self._env, **term_cfg.params)
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                if term_cfg.noise is not None and group_cfg.enable_corruption:
                    value = term_cfg.noise.apply(value)
                if term_cfg.scale is not None:
                    value = value * term_cfg.scale
                if term_cfg.clip is not None:
                    value = value.clamp(term_cfg.clip[0], term_cfg.clip[1])
                tensors.append(value)
            if not tensors:
                out[group_name] = torch.zeros(self.num_envs, 0, device=self.device)
                continue
            frame = (
                torch.cat(tensors, dim=-1)
                if group_cfg.concatenate_terms
                else torch.stack(tensors, dim=-1)
            )
            if group_cfg.history_length > 1:
                frame = self._stack_history(group_name, frame, group_cfg.history_length)
            out[group_name] = frame
        # Cache dimensions for downstream wrappers
        self._group_dims = {g: [int(t.shape[-1]) for t in (out[g],)] for g in out}
        return out

    def _stack_history(self, group_name: str, frame: torch.Tensor, h: int) -> torch.Tensor:
        """Roll ``frame`` into the group's history and return it flattened oldest->newest.

        On the first compute (or a num_envs / width change) the buffer is initialised by
        backfilling the current frame across all ``h`` slots — no zero padding. Each
        subsequent compute drops the oldest slot and appends ``frame``; envs that were just
        reset (``episode_length_buf == 0``, set in ``_reset_idx`` before this runs) backfill
        instead of rolling, so a fresh episode never carries stale pre-reset frames.
        """
        buf = self._history.get(group_name)
        if buf is None or buf.shape[0] != frame.shape[0] or buf.shape[-1] != frame.shape[-1]:
            buf = frame.unsqueeze(1).repeat(1, h, 1)
        else:
            rolled = torch.cat([buf[:, 1:], frame.unsqueeze(1)], dim=1)
            backfilled = frame.unsqueeze(1).expand(-1, h, -1)
            fresh = (self._env.episode_length_buf == 0).view(-1, 1, 1)
            buf = torch.where(fresh, backfilled, rolled)
        self._history[group_name] = buf
        return buf.reshape(frame.shape[0], -1)

    def group_obs_dim(self, group_name: str) -> int:
        # Lazy compute if not already populated
        if group_name not in self._group_dims:
            self.compute()
        return int(sum(self._group_dims[group_name]))
