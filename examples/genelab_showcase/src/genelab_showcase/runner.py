"""Shared play-only runner used by every showcase task.

The runner owns the env lifecycle, drives a fixed-length scripted-action loop, and
calls the subclass-supplied :py:meth:`_dump` hook every ``log_interval`` steps so each
showcase can render its own visual / textual evidence (PNG dump, histogram print,
joint-tracking error, ...).

Subclasses override :py:meth:`_scripted_action` to emit the per-step action tensor and
:py:meth:`_dump` to extract sensor / curriculum / robot state. The base class handles
device placement, log-directory creation, and viewer ticking — none of that should
duplicate per showcase.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from genelab.cache import ensure_project_cache
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

if TYPE_CHECKING:
    pass


class ShowcaseRunner:
    """Base class for the six play-only showcases.

    The runner builds a :class:`ManagerBasedRlEnv` from ``env_cfg``, then drives
    ``num_steps`` calls to :py:meth:`env.step` with the action returned from
    :py:meth:`_scripted_action`. Every ``log_interval`` steps it calls
    :py:meth:`_dump` so each subclass can write whatever evidence is relevant for its
    feature (PNG frames, histograms, tracking error, ...).

    ``log_root`` is created lazily on first dump. The default lives under
    ``logs/showcase/<task_slug>/``.
    """

    task_slug: str = "showcase"

    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        *,
        num_steps: int | None = None,
        log_interval: int = 20,
    ) -> None:
        self.env_cfg = env_cfg
        # Default ``num_steps`` from ``env.simulation.steps`` so the CLI ``--steps``
        # override (which lands on ``env.simulation.steps``) actually drives the loop
        # length. Subclasses can still pin a hard default via ``num_steps=...``.
        self.num_steps = int(num_steps if num_steps is not None else env_cfg.simulation.steps)
        self.log_interval = max(1, int(log_interval))
        self._log_root: Path | None = None
        self._env: ManagerBasedRlEnv | None = None

    # ------------------------------------------------------------------ hooks

    def _scripted_action(self, env: ManagerBasedRlEnv, step: int) -> torch.Tensor:
        """Return the per-step action tensor of shape ``(num_envs, num_actions)``.

        Default: zero action (each `JointPositionActionCfg` with
        ``use_default_offset=True`` interprets this as "go to ``default_joint_pos``").
        """

        return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    def _dump(self, env: ManagerBasedRlEnv, step: int) -> None:
        """Subclass hook — called every ``log_interval`` steps. Default: no-op."""

    # ------------------------------------------------------------------ utilities

    def log_root(self) -> Path:
        """Lazily create and return the per-showcase log directory."""

        if self._log_root is None:
            root = Path("logs") / "showcase" / self.task_slug
            root.mkdir(parents=True, exist_ok=True)
            self._log_root = root
        return self._log_root

    # ------------------------------------------------------------------ play

    def play(self) -> None:
        """Build the env, run the scripted loop, close cleanly.

        The kernel sets :py:attr:`env.viewer_closed` when the user closes the
        Genesis viewer; the loop polls that flag and breaks. No need to catch
        ``GenesisException`` here.
        """

        ensure_project_cache()
        env = ManagerBasedRlEnv(self.env_cfg)
        self._env = env
        last_step = self.num_steps
        try:
            env.reset()
            for step in range(self.num_steps):
                action = self._scripted_action(env, step)
                env.step(action)
                if env.viewer_closed:
                    last_step = step
                    break
                if step % self.log_interval == 0:
                    self._dump(env, step)
            # Always dump one final frame so the last state is captured.
            self._dump(env, last_step)
        finally:
            env.close()
            self._env = None
