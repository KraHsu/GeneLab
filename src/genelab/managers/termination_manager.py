"""Termination manager: computes done / time_out signals from term outputs."""

from dataclasses import dataclass

import torch

from genelab.managers._base import BaseTermManager
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    time_out: bool = False


class TerminationManager(BaseTermManager[TerminationTermCfg]):
    # Inherits __init__ from BaseTermManager — no extra constructor state needed.

    def _post_init(self) -> None:
        """Allocate per-term done bool buffers and the two aggregate (terminated / truncated) bufs."""
        self._term_dones: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in self._term_names
        }
        self._truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)

    @property
    def dones(self) -> torch.Tensor:
        return self._truncated_buf | self._terminated_buf

    @property
    def time_outs(self) -> torch.Tensor:
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        names = list(self._term_dones.keys())
        if not names:
            return {}
        # One CUDA sync for all termination terms, instead of one per term.
        counts = torch.stack(
            [torch.count_nonzero(self._term_dones[name][env_ids]) for name in names]
        ).tolist()
        return {f"Episode_Termination/{name}": float(c) for name, c in zip(names, counts)}

    def compute(self) -> torch.Tensor:
        self._truncated_buf.zero_()
        self._terminated_buf.zero_()
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            value = term_cfg.func(self._env, **term_cfg.params)
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            self._term_dones[name][:] = value
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> torch.Tensor:
        return self._term_dones[name]
