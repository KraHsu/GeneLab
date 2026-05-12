"""Environment registry namespace for downstream GeneLab extensions.

Keep project-specific environments in your own Python package and register them with
``genelab.registry.register_env`` or ``genelab.lab.ENVS``.
"""

from genelab.configs import ManagerBasedEnvCfg
from genelab.lab import ManagerBasedEnv
from genelab.registry import ENVS, RegistryEntry, register_env

__all__ = [
    "ENVS",
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "RegistryEntry",
    "register_env",
]
