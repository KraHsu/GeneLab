"""Environment registry namespace for downstream GeneLab extensions.

Keep project-specific environments in your own Python package and register them with
``genelab.registry.register_env`` or ``genelab.lab.ENVS``.
"""

from genelab.configs import ManagerBasedEnvCfg
from genelab.entity import RobotState  # re-export; canonical home is genelab.entity
from genelab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    RobotEntityCfg,
)
from genelab.lab import ManagerBasedEnvProtocol
from genelab.registry import ENVS, RegistryEntry, register_env

__all__ = [
    "ENVS",
    "ManagerBasedEnvCfg",
    "ManagerBasedEnvProtocol",
    "ManagerBasedRlEnv",
    "ManagerBasedRlEnvCfg",
    "RegistryEntry",
    "RobotEntityCfg",
    "RobotState",
    "register_env",
]
