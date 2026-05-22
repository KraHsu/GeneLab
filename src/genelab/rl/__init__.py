"""RL runner, backend abstraction, and VecEnv integration."""

from genelab.rl.backends import Backend, select_backend
from genelab.rl.config import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from genelab.rl.profiler import maybe_profile, profiler_enabled
from genelab.rl.runner import AgentKind, play_task, train_task
from genelab.rl.sb3_config import Sb3AgentCfg, Sb3ExperimentCfg, Sb3HerCfg, Sb3PolicyCfg
from genelab.rl.skrl_config import SkrlAgentCfg, SkrlExperimentCfg, SkrlModelCfg
from genelab.rl.vecenv import VecEnvBase

__all__ = [
    "AgentKind",
    "Backend",
    "RslRlBaseRunnerCfg",
    "RslRlModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "Sb3AgentCfg",
    "Sb3ExperimentCfg",
    "Sb3HerCfg",
    "Sb3PolicyCfg",
    "SkrlAgentCfg",
    "SkrlExperimentCfg",
    "SkrlModelCfg",
    "VecEnvBase",
    "maybe_profile",
    "play_task",
    "profiler_enabled",
    "select_backend",
    "train_task",
]


def __getattr__(name: str) -> object:
    # Deprecated single-backend re-export (ADR-0007 / R6). The vecenv adapters now
    # live under ``genelab.rl.vecenvs.<lib>``; this kept the asymmetric top-level
    # ``RslRlVecEnvWrapper`` export working for one release. Removed from ``__all__``.
    if name == "RslRlVecEnvWrapper":
        import warnings

        from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper as _cls

        warnings.warn(
            "genelab.rl.RslRlVecEnvWrapper is deprecated; import it from "
            "genelab.rl.vecenvs.rsl_rl instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
