"""Typed backend-config dispatch (ADR-0016 step 2).

Every config the backend registry dispatches on subclasses ``BackendConfig``, and
``select_backend`` resolves each to its backend. The config imports are RL-library
free (dataclasses only); ``select_backend`` triggers the backend modules' import,
which guard their skrl / SB3 imports behind function-local imports.
"""

from genelab.rl.backends.sb3.config import Sb3AgentCfg
from genelab.rl.backends.skrl.config import SkrlAgentCfg
from genelab.rl.config import BackendConfig, RslRlOnPolicyRunnerCfg

# The three configs the registry keys on (one per bundled backend).
DISPATCH_CONFIGS = (RslRlOnPolicyRunnerCfg, SkrlAgentCfg, Sb3AgentCfg)


def test_dispatch_configs_subclass_backend_config() -> None:
    for cls in DISPATCH_CONFIGS:
        assert issubclass(cls, BackendConfig), cls


def test_select_backend_resolves_each_config() -> None:
    """Each config resolves to its backend, whose ``cfg_type`` is that config class."""
    from genelab.rl.backends import select_backend

    for cls, name in (
        (RslRlOnPolicyRunnerCfg, "rsl_rl"),
        (SkrlAgentCfg, "skrl"),
        (Sb3AgentCfg, "sb3"),
    ):
        backend = select_backend(cls())
        assert backend.name == name
        assert issubclass(backend.cfg_type, BackendConfig)
