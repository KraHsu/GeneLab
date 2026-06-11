"""Training-only action perturbation in the ActionManager (sim2real hardening).

Two opt-in knobs, both default-off (no behavior change when unset) and never exported
into the deployed policy — they only perturb what is *applied* in sim during training so
the policy learns to tolerate a noisy, latent control loop:

* ``action_noise_std`` — per-step Gaussian noise added to the applied action.
* ``action_delay_steps`` — per-env integer latency (sampled at reset) so the command that
  reaches the actuators is a few control steps stale.

The raw policy action still flows to ``last_action`` observations / action-rate metrics
unperturbed (the network's own output), matching the real deploy where the policy sees
what it emitted, not what the wire delivered.
"""

import torch

from genelab.managers.action_manager import ActionManager, ActionTerm, ActionTermCfg


class _RecordingTerm(ActionTerm):
    """Records the (perturbed) action slice handed to ``process_actions``."""

    def __init__(self, cfg: "ActionTermCfg", env) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg, env)
        self.received: torch.Tensor | None = None
        self._dim = cfg.dim  # type: ignore[attr-defined]

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self.received if self.received is not None else torch.zeros(1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self.received = actions.clone()

    def apply_actions(self) -> None: ...


class _TermCfg(ActionTermCfg):
    def __init__(self, dim: int) -> None:
        super().__init__(class_type=_RecordingTerm)
        self.dim = dim


class _FakeEnv:
    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device


def test_no_perturbation_applies_raw_action() -> None:
    env = _FakeEnv(num_envs=2)
    mgr = ActionManager({"t": _TermCfg(dim=3)}, env)
    action = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mgr.process_action(action)
    term = mgr._terms["t"]
    assert torch.equal(term.received, action)  # type: ignore[arg-type]
    assert torch.equal(mgr.action, action)


def test_action_noise_perturbs_applied_but_not_obs_action() -> None:
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=64)
    mgr = ActionManager({"t": _TermCfg(dim=3)}, env, action_noise_std=0.2)
    action = torch.zeros(64, 3)
    mgr.process_action(action)
    term = mgr._terms["t"]
    # Applied action is noised away from zero...
    assert term.received.abs().mean().item() > 0.0  # type: ignore[union-attr]
    # ...but the cached policy action (feeds last_action obs / action-rate) stays raw.
    assert torch.equal(mgr.action, action)


def test_action_delay_applies_stale_command() -> None:
    env = _FakeEnv(num_envs=2)
    # Constant 1-step latency so the assertion is deterministic.
    mgr = ActionManager({"t": _TermCfg(dim=2)}, env, action_delay_steps=(1, 1))
    mgr.reset()  # sample the per-env delay, clear the buffer to zeros
    term = mgr._terms["t"]

    a0 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mgr.process_action(a0)
    # 1-step delay: the very first command is held back; actuators still see the zero prefill.
    assert torch.equal(term.received, torch.zeros(2, 2))  # type: ignore[arg-type]

    a1 = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    mgr.process_action(a1)
    # Now the actuators receive the previous step's command, not the latest.
    assert torch.equal(term.received, a0)  # type: ignore[arg-type]
