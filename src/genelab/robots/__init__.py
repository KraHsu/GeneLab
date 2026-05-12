"""Robot registry namespace for downstream GeneLab extensions.

Keep project-specific robot implementations in your own Python package and register them with
``genelab.registry.register_robot`` or ``genelab.lab.ROBOTS``.
"""

from genelab.registry import ROBOTS, RegistryEntry, register_robot

__all__ = [
    "ROBOTS",
    "RegistryEntry",
    "register_robot",
]
