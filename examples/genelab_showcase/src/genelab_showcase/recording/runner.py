"""Recording-showcase runner: scripted joint-1 sweep; recording fires automatically."""

import math
from typing import TYPE_CHECKING

import torch

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class RecordingShowcaseRunner(ShowcaseRunner):
    """Wave Franka joint 1 ±0.5 rad on a 4-second period; the recordings on the env
    cfg do all the data capture. No ``_dump`` is needed — the live plots and file
    writers run on Genesis's recorder threads."""

    task_slug = "recording"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        # ``log_interval`` is unused (we don't override ``_dump``) but the base class
        # requires a value.
        super().__init__(env_cfg, log_interval=20)
        self._joint1_idx: int | None = None

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        if self._joint1_idx is None:
            self._joint1_idx = env.articulations["robot"].joint_names.index("joint1")
        # period ≈ 4 s at dt=0.02 (decimation 2 × physics 0.01) → 200 control steps.
        phase = 2.0 * math.pi * step / 200.0
        action[:, self._joint1_idx] = math.sin(phase) * 0.5
        return action
