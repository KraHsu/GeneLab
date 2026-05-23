"""Public extension API for GeneLab.

Third-party packages register robots, environments, tasks, and RL backends via
the symbols re-exported here. The implementations live in :mod:`genelab.registry`
(the first three kinds + the :class:`Runnable` task contract) and
:mod:`genelab.rl.backends` (the fourth). This module is the stable import path —
the implementation modules may evolve (ADR-0008 / ROADMAP §9 R7).

Opt-in mechanisms: the ``genelab.extensions`` entry-point group (auto-loaded) and
the ``genelab --import MODULE`` CLI flag (one-off).
"""

from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    Runnable,
    register_env,
    register_robot,
    register_task,
)
from genelab.rl.backends import Backend, register_backend

__all__ = [
    "ENVS",
    "ROBOTS",
    "TASKS",
    "Backend",
    "Runnable",
    "register_backend",
    "register_env",
    "register_robot",
    "register_task",
]
