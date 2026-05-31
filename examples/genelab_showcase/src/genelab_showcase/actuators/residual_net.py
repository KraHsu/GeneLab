"""Tiny TorchScript residual net for the MLP-residual actuator showcase.

``MlpResidualActuator`` loads a TorchScript module from ``network_file`` whose forward
maps ``[target_pos - joint_pos, joint_vel]`` (last dim) to a per-joint residual torque.
This module generates a deterministic ``nn.Linear(2, 1)`` that acts as a tracking
*compensator*: ``residual = 80 * pos_error``. The positive position-error weight pushes extra
torque toward the target, recovering tracking the base PD alone loses as its gains drop — so
the residual always tightens tracking and the benefit grows as the kp slider is lowered. Sized
to stay well inside the arm's 87 Nm effort budget at the demo's amplitudes, so it sharpens
tracking without overshoot or instability. The showcase overlays this arm against a plain
IdealPD arm so the residual's effect is visible.

The net is written under ``CACHE_DIR`` (git-ignored) on first use rather than committed
as a binary; the env cfg and the unit test both resolve it through
:func:`residual_net_path`.
"""

from typing import TYPE_CHECKING

from genelab.utils.paths import CACHE_DIR

if TYPE_CHECKING:
    import torch

# residual = w · [pos_error, joint_vel] + b: a position-error compensator (extra effort toward
# the target) on top of the DC-motor torque. Velocity weight 0 — pure stiffness compensation,
# so the residual never adds phase lag, only tightens tracking.
_WEIGHT: tuple[float, float] = (80.0, 0.0)
_BIAS: float = 0.0


def _build_residual_net() -> "torch.jit.ScriptModule":
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
