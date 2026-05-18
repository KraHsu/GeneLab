"""Unit tests for ``UniformVelocityCommand`` env-group fields (P2).

Tests use a minimal fake env (no Genesis) and instantiate the term directly. The four
env groups — standing / heading / forward / world — are sampled independently per
resample, so each test exercises one ratio at a time and lets the law of large numbers
handle Bernoulli noise (``num_envs=4096`` keeps ``binomial std / p`` under ~1.5 %).
"""

from dataclasses import dataclass

import torch

from genelab.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


# --------------------------------------------------------------------- fakes


@dataclass
class _FakeRobotState:
    root_quat: torch.Tensor  # (B, 4) wxyz


class _FakeEnv:
    def __init__(self, num_envs: int, *, yaw: float = 0.0, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        # Build root_quat from a single shared yaw so heading-error math is predictable.
        c = torch.cos(torch.tensor(yaw / 2))
        s = torch.sin(torch.tensor(yaw / 2))
        quat = torch.zeros(num_envs, 4, device=device)
        quat[:, 0] = c  # w
        quat[:, 3] = s  # z
        self.robot_state = _FakeRobotState(root_quat=quat)


def _build(
    cfg: UniformVelocityCommandCfg, *, num_envs: int, yaw: float = 0.0
) -> UniformVelocityCommand:
    env = _FakeEnv(num_envs=num_envs, yaw=yaw)
    return UniformVelocityCommand(cfg, env)


# --------------------------------------------------------------------- backward compatibility


def test_default_cfg_matches_pre_p2_heading_command_behaviour() -> None:
    """With only ``heading_command=True`` set, every non-standing env must be a heading env.

    This is the contract callers relied on before P2: passing no env-group ratios should
    not change anything. Defaults: standing=0, heading=1, forward=0, world=0.
    """
    torch.manual_seed(0)
    cfg = UniformVelocityCommandCfg(heading_command=True)
    term = _build(cfg, num_envs=2048)
    env_ids = torch.arange(2048)
    term._resample_command(env_ids)
    assert term.is_standing_env.float().mean().item() == 0.0
    assert term.is_forward_env.float().mean().item() == 0.0
    assert term.is_world_env.float().mean().item() == 0.0
    # Every env is a heading env at rel_heading_envs=1.0.
    assert term.is_heading_env.all().item()


def test_legacy_standing_ratio_preserved() -> None:
    """``rel_standing_envs=0.3`` alone (no other new fields) keeps roughly 30 % standing."""
    torch.manual_seed(0)
    cfg = UniformVelocityCommandCfg(rel_standing_envs=0.3, heading_command=True)
    term = _build(cfg, num_envs=4096)
    term._resample_command(torch.arange(4096))
    standing_ratio = term.is_standing_env.float().mean().item()
    assert abs(standing_ratio - 0.3) < 0.02
    # Standing envs must have all-zero commands.
    standing_ids = term.is_standing_env.nonzero(as_tuple=False).flatten()
    assert torch.all(term.command[standing_ids] == 0.0)


# --------------------------------------------------------------------- standing


def test_standing_envs_zero_command_through_update() -> None:
    """``_update_command`` must keep standing envs at zero even after heading PD runs."""
    torch.manual_seed(1)
    cfg = UniformVelocityCommandCfg(rel_standing_envs=0.5, heading_command=True)
    term = _build(cfg, num_envs=512)
    term._resample_command(torch.arange(512))
    term._update_command()
    standing_ids = term.is_standing_env.nonzero(as_tuple=False).flatten()
    assert torch.all(term.command[standing_ids] == 0.0)


# --------------------------------------------------------------------- heading


def test_heading_envs_get_pd_driven_ang_vel_z() -> None:
    """For heading envs, ``ang_vel_z`` is recomputed from heading error each step."""
    torch.manual_seed(2)
    cfg = UniformVelocityCommandCfg(
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
    )
    # Body yaw = 0; sampled heading targets are uniform on (-pi, pi). PD output should
    # equal clamp(0.5 * heading_error, -1, 1) for every env.
    term = _build(cfg, num_envs=256, yaw=0.0)
    term._resample_command(torch.arange(256))
    # Pre-update ang_vel_z is the raw sampled value; after _update_command it's the PD.
    term._update_command()
    expected = torch.clamp(
        cfg.heading_control_stiffness * term._heading_target,
        cfg.ranges.ang_vel_z[0],
        cfg.ranges.ang_vel_z[1],
    )
    assert torch.allclose(term.command[:, 2], expected, atol=1e-6)


def test_non_heading_envs_keep_sampled_ang_vel_z() -> None:
    """With ``rel_heading_envs=0`` and ``heading_command=True``, ωz keeps the raw sample."""
    torch.manual_seed(3)
    cfg = UniformVelocityCommandCfg(rel_heading_envs=0.0, heading_command=True)
    term = _build(cfg, num_envs=256)
    term._resample_command(torch.arange(256))
    sampled_omega_z = term.command[:, 2].clone()
    term._update_command()
    assert torch.allclose(term.command[:, 2], sampled_omega_z)
    assert not term.is_heading_env.any().item()


def test_heading_ratio_matches_configured_proportion() -> None:
    torch.manual_seed(4)
    cfg = UniformVelocityCommandCfg(rel_heading_envs=0.3, heading_command=True)
    term = _build(cfg, num_envs=4096)
    term._resample_command(torch.arange(4096))
    ratio = term.is_heading_env.float().mean().item()
    assert abs(ratio - 0.3) < 0.02


def test_heading_command_false_disables_all_heading_envs() -> None:
    """When ``heading_command=False``, no env is ever a heading env regardless of ratio."""
    torch.manual_seed(5)
    cfg = UniformVelocityCommandCfg(rel_heading_envs=1.0, heading_command=False)
    term = _build(cfg, num_envs=256)
    term._resample_command(torch.arange(256))
    assert not term.is_heading_env.any().item()


# --------------------------------------------------------------------- forward


def test_forward_envs_rewrite_command_to_strict_forward_motion() -> None:
    """Forward envs: ``vx >= 0.3``, ``vy == 0``, ``ωz == 0`` after resample."""
    torch.manual_seed(6)
    cfg = UniformVelocityCommandCfg(rel_forward_envs=1.0, heading_command=False)
    term = _build(cfg, num_envs=256)
    term._resample_command(torch.arange(256))
    assert term.is_forward_env.all().item()
    assert torch.all(term.command[:, 0] >= 0.3)
    assert torch.all(term.command[:, 1] == 0.0)
    assert torch.all(term.command[:, 2] == 0.0)


def test_forward_ratio_matches_configured_proportion() -> None:
    torch.manual_seed(7)
    cfg = UniformVelocityCommandCfg(rel_forward_envs=0.2, heading_command=True)
    term = _build(cfg, num_envs=4096)
    term._resample_command(torch.arange(4096))
    ratio = term.is_forward_env.float().mean().item()
    assert abs(ratio - 0.2) < 0.02


# --------------------------------------------------------------------- world


def test_world_envs_rotate_world_velocity_into_body_frame() -> None:
    """Body yaw=π/2 ⇒ a (1, 0, 0) world cmd should read as (0, -1, 0) in body frame.

    The rotation in :meth:`_update_command` is ``[cos φ, sin φ; -sin φ, cos φ]`` applied
    to the stored world-frame ``(vx_w, vy_w)``. For yaw=π/2 that gives [0, 1; -1, 0],
    so ``(1, 0) -> (0, -1)``.
    """
    cfg = UniformVelocityCommandCfg(rel_world_envs=1.0, heading_command=False)
    term = _build(cfg, num_envs=1, yaw=torch.pi / 2)
    # Force a known world-frame sample so the rotation is unambiguous.
    term._command[0] = torch.tensor([1.0, 0.0, 0.0])
    term._command_w[0] = torch.tensor([1.0, 0.0, 0.0])
    term._is_world[0] = True
    term._update_command()
    assert torch.allclose(term.command[0, :2], torch.tensor([0.0, -1.0]), atol=1e-6)


def test_world_ratio_matches_configured_proportion() -> None:
    torch.manual_seed(8)
    cfg = UniformVelocityCommandCfg(rel_world_envs=0.25, heading_command=True)
    term = _build(cfg, num_envs=4096)
    term._resample_command(torch.arange(4096))
    ratio = term.is_world_env.float().mean().item()
    assert abs(ratio - 0.25) < 0.02


# --------------------------------------------------------------------- precedence


def test_standing_overrides_forward_when_both_fire() -> None:
    """Independent Bernoullis can both pick the same env; standing must win after resample."""
    cfg = UniformVelocityCommandCfg(rel_standing_envs=1.0, rel_forward_envs=1.0)
    term = _build(cfg, num_envs=128)
    term._resample_command(torch.arange(128))
    # Every env is both standing and forward; standing's zero override is final.
    assert term.is_standing_env.all().item()
    assert term.is_forward_env.all().item()
    assert torch.all(term.command == 0.0)


def test_standing_overrides_heading_during_update() -> None:
    """Standing+heading env: heading PD writes ωz, then standing zero-out runs after."""
    cfg = UniformVelocityCommandCfg(
        rel_standing_envs=1.0, rel_heading_envs=1.0, heading_command=True
    )
    term = _build(cfg, num_envs=64)
    term._resample_command(torch.arange(64))
    term._update_command()
    assert torch.all(term.command == 0.0)


# --------------------------------------------------------------------- partial-resample


def test_resample_subset_keeps_other_envs_state() -> None:
    """Calling ``_resample_command`` on a subset of env_ids must not touch other envs."""
    torch.manual_seed(9)
    cfg = UniformVelocityCommandCfg(rel_heading_envs=1.0)
    term = _build(cfg, num_envs=8)
    term._resample_command(torch.arange(8))
    before = term.command.clone()
    headings_before = term._heading_target.clone()
    resample_ids = torch.tensor([1, 3, 5])
    term._resample_command(resample_ids)
    # Untouched envs preserve their command + heading target.
    untouched = torch.tensor([0, 2, 4, 6, 7])
    assert torch.allclose(term.command[untouched], before[untouched])
    assert torch.allclose(term._heading_target[untouched], headings_before[untouched])


def test_set_external_source_pins_term_for_bridge_writes() -> None:
    """External drivers (teleop bridges) need ``_resample_command`` to stop
    re-randomising and ``_update_command`` to be a no-op so per-step writes to
    ``_command`` survive both ``compute()`` and any subsequent env reset.
    """
    cfg = UniformVelocityCommandCfg(
        rel_standing_envs=0.5,
        rel_heading_envs=0.5,
        rel_forward_envs=0.2,
        rel_world_envs=0.2,
        heading_command=True,
    )
    term = _build(cfg, num_envs=1)
    term._resample_command(torch.arange(1))

    term.set_external_source(enabled=True)

    # Command zeroed so the first obs the policy reads matches slider default.
    assert torch.allclose(term.command, torch.zeros(1, 3))
    # Group flags cleared so _update_command becomes a no-op.
    assert not term._is_heading.any()
    assert not term._is_standing.any()
    assert not term._is_forward.any()
    assert not term._is_world.any()
    # Cfg is left untouched — pinning is governed by the runtime flag, not by
    # mutating user-owned configuration.
    assert term.cfg.heading_command is True
    assert term.cfg.rel_heading_envs == 0.5

    # An external write into _command must survive a compute(dt) call.
    term._command[:] = torch.tensor([[0.5, -0.2, 0.3]])
    term.compute(dt=0.02)
    assert torch.allclose(term.command, torch.tensor([[0.5, -0.2, 0.3]]))

    # A reset that fires _resample must NOT re-randomize while pinned.
    term._time_left[:] = -1.0  # force the next compute to call _resample
    term.compute(dt=0.02)
    assert torch.allclose(term.command, torch.tensor([[0.5, -0.2, 0.3]]))

    # Releasing the external source restores normal sampling behavior.
    term.set_external_source(enabled=False)
    term._resample_command(torch.arange(1))
    # _command should now reflect a fresh sample (very unlikely to be exactly the pinned tuple).
    assert not torch.allclose(term.command, torch.tensor([[0.5, -0.2, 0.3]]))
