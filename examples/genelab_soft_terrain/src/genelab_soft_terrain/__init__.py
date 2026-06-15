"""GeneLab example: legged locomotion over analytic deformable (soft) terrain.

Stage 0 of ADR-0001 — the robot is supported entirely by an analytic compliance force
over a virtual surface (no rigid floor under the feet). This is a *stand-and-sink* demo:
without a tangential/friction term (stage 1) the robot holds a stance and settles at its
sinkage equilibrium, it does not yet walk.
"""

from genelab_soft_terrain.g1_mattress import g1_mattress_velocity_env_cfg
from genelab_soft_terrain.go1_soft_env_cfg import go1_soft_stand_env_cfg
from genelab_soft_terrain.go1_soft_velocity_env_cfg import go1_soft_velocity_env_cfg

__all__ = ["g1_mattress_velocity_env_cfg", "go1_soft_stand_env_cfg", "go1_soft_velocity_env_cfg"]
