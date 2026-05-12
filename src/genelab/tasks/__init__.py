"""Task registry namespace for downstream GeneLab extensions.

Keep project-specific task definitions in your own Python package and register them with
``genelab.registry.register_task`` or ``genelab.lab.TASKS``.
"""

from genelab.configs import TaskCfg
from genelab.registry import TASKS, RegistryEntry, register_task

__all__ = [
    "TASKS",
    "RegistryEntry",
    "TaskCfg",
    "register_task",
]
