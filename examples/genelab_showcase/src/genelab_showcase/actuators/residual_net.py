"""Tiny TorchScript residual net for the MLP-residual actuator showcase.

``MlpResidualActuator`` loads a TorchScript module from ``network_file`` whose forward
maps ``[target_pos - joint_pos, joint_vel]`` (last dim) to a per-joint residual torque.
This module generates a deterministic ``nn.Linear(2, 1)`` that applies a small
velocity-damping correction (``residual = -0.05 * joint_vel``) — large enough to make
the residual code path observable, tiny relative to the arm's 87 Nm effort budget so
it cannot destabilise the demo.

The net is written under ``CACHE_DIR`` (git-ignored) on first use rather than committed
as a binary; the env cfg and the unit test both resolve it through
:func:`residual_net_path`.
"""

from genelab.utils.paths import CACHE_DIR

# residual = w · [pos_error, joint_vel] + b. A small negative velocity weight adds a
# little extra joint damping on top of the DC-motor torque.
_WEIGHT: tuple[float, float] = (0.0, -0.05)
_BIAS: float = 0.0


def _build_residual_net() -> "object":
    import torch

    net = torch.nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([_WEIGHT]))
        net.bias.copy_(torch.tensor([_BIAS]))
    net.eval()
    return torch.jit.script(net)


def residual_net_path() -> str:
    """Materialize the showcase residual net under ``CACHE_DIR`` and return its path.

    Always (re)writes, so changing ``_WEIGHT`` / ``_BIAS`` can never leave a stale
    cached net behind. The net is tiny and deterministic, so the rewrite is cheap.
    """
    import torch

    path = CACHE_DIR / "showcase" / "mlp_residual_net.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(_build_residual_net(), str(path))
    return str(path)
