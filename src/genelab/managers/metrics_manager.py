"""Metrics manager: accumulates per-step diagnostic values, reports episode means.

Parallel to :class:`~genelab.managers.reward_manager.RewardManager` but stripped of
the things that don't make sense for diagnostics:

* No weights — metrics aren't part of the training loss.
* No ``dt`` scaling — the episode mean is over control steps, not seconds.
* No clamping / nan-fill — if a metric goes NaN you want to see it.

Reference parity for ``MetricsManager``. We skip the reference's ``per_substep`` flag (we
don't have a substep hook in GeneLab's step loop) and the ``reduce="last"``
option (no current consumer). Add either back if needed — the manager's
internal layout was designed with those extensions in mind.
"""

from dataclasses import dataclass

import torch

from genelab.managers._base import BaseTermManager
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg


@dataclass
class MetricsTermCfg(ManagerTermBaseCfg):
    """Configuration for a metrics term.

    No extra fields beyond :class:`ManagerTermBaseCfg` for now — the value is in
    the manager's calling convention (no weight / no dt scaling / no nan-fill)
    rather than per-term knobs.
    """


class MetricsManager(BaseTermManager[MetricsTermCfg]):
    """Aggregates per-step diagnostic metrics; reports per-env means at reset.

    Per-env step counter tracks how many compute() calls happened per env. At
    ``reset`` the manager returns ``Episode_Metrics/<name> = mean(sum / count)``
    across the reset envs, then zeros the accumulators for those envs only.
    """

    def _post_init(self) -> None:
        """Allocate per-term episode-sum buffers and the per-env step counter."""
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self._term_names
        }
        # Per-env compute() call count, used to turn accumulated sums into means at
        # reset. Stays in sync with episode_length because compute() is invoked once
        # per env step alongside reward_manager.compute().
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def compute(self) -> None:
        """Evaluate every term and accumulate into the per-env episode sums."""
        if not self._term_names:
            return
        self._step_count += 1
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            value = term_cfg.func(self._env, **term_cfg.params)
            self._episode_sums[name] += value

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        """Pop per-env episode means for the reset envs as ``Episode_Metrics/<name>`` dict.

        Mirrors :meth:`RewardManager.reset` so the env's ``_reset_idx`` can merge
        the dict directly into ``extras["log"]``. Envs that hit reset without any
        compute() calls (e.g. the very first reset) get a zero count guarded by a
        ``clamp(min=1)`` so the division stays finite.
        """
        if env_ids is None:
            env_ids = slice(None)
        names = list(self._episode_sums.keys())
        if not names:
            return {}
        # Pull all means through a single host sync — the same single-stack pattern as
        # RewardManager.reset uses to keep per-step CUDA syncs to one.
        counts = self._step_count[env_ids].float().clamp(min=1.0)
        means = torch.stack(
            [torch.mean(self._episode_sums[name][env_ids] / counts) for name in names]
        )
        means_list = means.tolist()
        for name in names:
            self._episode_sums[name][env_ids] = 0.0
        self._step_count[env_ids] = 0
        return {f"Episode_Metrics/{name}": value for name, value in zip(names, means_list)}
