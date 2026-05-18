"""Unit tests for ``genelab.mdp.dr`` events.

Uses fake robots that record Genesis-setter calls so we can verify the DR
functions translate cfg ranges and link/joint filters into the right tensor
shapes, link indices, and env-id passthroughs without needing a Genesis runtime.

End-to-end encoder-bias coverage builds a fake env: ``JointPositionAction``
subtracts the bias from its target so the physical joint sits ``bias`` away
from the commanded reference, while ``mdp.joint_pos_rel`` returns the raw
``joint_pos − default`` (no bias) so the policy actually sees the perturbation.
"""

from dataclasses import dataclass, field

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp import dr
from genelab.mdp.observations import joint_pos_rel


def _link_cfg(env, *names: str) -> SceneEntityCfg:
    cfg = SceneEntityCfg(name="robot", link_names=names or None)
    cfg.resolve(env)
    return cfg


def _joint_cfg(env, *names: str) -> SceneEntityCfg:
    cfg = SceneEntityCfg(name="robot", joint_names=names or None)
    cfg.resolve(env)
    return cfg


# --------------------------------------------------------------------- fakes


@dataclass
class _SetterCall:
    """One captured call to a Genesis-style setter."""

    tensor: torch.Tensor
    links_idx_local: list[int] | None
    envs_idx: torch.Tensor | None


class _RecordingRobot:
    """Captures the most recent set_* call's args so tests can introspect."""

    def __init__(self) -> None:
        self.friction: _SetterCall | None = None
        self.com_shift: _SetterCall | None = None
        self.mass_shift: _SetterCall | None = None

    def set_friction_ratio(self, friction, links_idx_local=None, envs_idx=None) -> None:
        self.friction = _SetterCall(friction, links_idx_local, envs_idx)

    def set_COM_shift(self, com_shift, links_idx_local=None, envs_idx=None) -> None:
        self.com_shift = _SetterCall(com_shift, links_idx_local, envs_idx)

    def set_mass_shift(self, mass_shift, links_idx_local=None, envs_idx=None) -> None:
        self.mass_shift = _SetterCall(mass_shift, links_idx_local, envs_idx)


class _StubRobotState:
    """Minimal surface for encoder_bias tests."""

    def __init__(self, num_envs: int, num_dofs: int) -> None:
        self.encoder_bias = torch.zeros(num_envs, num_dofs)
        self.joint_pos = torch.zeros(num_envs, num_dofs)


@dataclass
class _FakeEnv:
    num_envs: int
    link_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    device: str = "cpu"
    robot: _RecordingRobot = field(default_factory=_RecordingRobot)
    robot_state: _StubRobotState | None = None
    default_joint_pos: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.robot_state is None:
            self.robot_state = _StubRobotState(self.num_envs, len(self.joint_names) or 1)
        if self.default_joint_pos is None:
            self.default_joint_pos = torch.zeros(len(self.joint_names) or 1)


# --------------------------------------------------------------------- geom_friction


def test_geom_friction_shared_random_gives_one_value_per_env() -> None:
    """``shared_random=True``: every selected link of a given env gets the same sample."""
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=4, link_names=["base", "left_foot", "right_foot", "arm"])
    dr.geom_friction(
        env,
        env_ids=None,
        asset_cfg=_link_cfg(env, "left_foot", "right_foot"),
        ranges=(0.5, 1.5),
        shared_random=True,
    )
    call = env.robot.friction
    assert call is not None
    assert call.tensor.shape == (4, 2)
    # Per-env: the two feet share the same value (broadcast from one Bernoulli sample).
    assert torch.all(call.tensor[:, 0] == call.tensor[:, 1])
    assert call.links_idx_local == [1, 2]
    assert call.envs_idx is not None
    assert torch.equal(call.envs_idx, torch.arange(4))


def test_geom_friction_independent_random_decorrelates_links() -> None:
    torch.manual_seed(1)
    env = _FakeEnv(num_envs=128, link_names=["base", "left_foot", "right_foot"])
    dr.geom_friction(
        env,
        env_ids=None,
        asset_cfg=_link_cfg(env, "left_foot", "right_foot"),
        ranges=(0.5, 1.5),
        shared_random=False,
    )
    call = env.robot.friction
    assert call is not None
    # With independent draws the two columns should disagree on most envs.
    disagreements = (call.tensor[:, 0] != call.tensor[:, 1]).sum().item()
    assert disagreements > 100


def test_geom_friction_respects_env_ids_subset() -> None:
    env = _FakeEnv(num_envs=8, link_names=["base", "foot"])
    subset = torch.tensor([2, 5, 7])
    dr.geom_friction(env, env_ids=subset, asset_cfg=_link_cfg(env, "foot"))
    call = env.robot.friction
    assert call is not None
    assert call.tensor.shape == (3, 1)
    assert torch.equal(call.envs_idx, subset)


def test_geom_friction_asset_cfg_unset_covers_every_link() -> None:
    """``SceneEntityCfg`` with no ``link_names`` falls back to every link in env."""
    env = _FakeEnv(num_envs=1, link_names=["a", "b", "c"])
    dr.geom_friction(env, env_ids=None, asset_cfg=SceneEntityCfg(name="robot"))
    call = env.robot.friction
    assert call is not None
    assert call.links_idx_local == [0, 1, 2]


def test_geom_friction_converts_absolute_to_ratio_via_nominal() -> None:
    """mjlab parity: the sampled value is the absolute friction, not a ratio.

    Genesis only exposes a batched ``set_friction_ratio`` setter, so we divide
    by each link's nominal friction (read from ``robot.links[i].geoms[0]._friction``)
    before passing the value through.
    """
    from dataclasses import dataclass

    @dataclass
    class _FakeGeom:
        _friction: float

    @dataclass
    class _FakeLink:
        geoms: list[_FakeGeom]

    class _RobotWithNominal(_RecordingRobot):
        # Two link slots: link 0 has nominal=0.6 (G1 foot default), link 1 has 1.0.
        links = [
            _FakeLink(geoms=[_FakeGeom(_friction=0.6)]),
            _FakeLink(geoms=[_FakeGeom(_friction=1.0)]),
        ]

    env = _FakeEnv(num_envs=1, link_names=["foot", "other"])
    env.robot = _RobotWithNominal()
    torch.manual_seed(7)
    # Constant range so the sampled absolute friction is exactly 0.9.
    dr.geom_friction(
        env,
        env_ids=None,
        asset_cfg=_link_cfg(env, "foot", "other"),
        ranges=(0.9, 0.9),
        shared_random=True,
    )
    call = env.robot.friction
    assert call is not None
    # ratio[link0] = 0.9 / 0.6 = 1.5; ratio[link1] = 0.9 / 1.0 = 0.9.
    assert torch.allclose(call.tensor[0, 0], torch.tensor(1.5), atol=1e-6)
    assert torch.allclose(call.tensor[0, 1], torch.tensor(0.9), atol=1e-6)


# --------------------------------------------------------------------- body_com_offset


def test_body_com_offset_only_fills_axes_listed_in_ranges() -> None:
    """Axes absent from the ranges dict must stay at exactly zero."""
    torch.manual_seed(2)
    env = _FakeEnv(num_envs=64, link_names=["pelvis"])
    dr.body_com_offset(
        env,
        env_ids=None,
        asset_cfg=_link_cfg(env, "pelvis"),
        ranges={0: (-0.02, 0.02), 2: (-0.03, 0.03)},  # x and z only
    )
    call = env.robot.com_shift
    assert call is not None
    assert call.tensor.shape == (64, 1, 3)
    assert torch.all(call.tensor[..., 1] == 0.0)  # y untouched
    assert (call.tensor[..., 0].abs() > 0).any()
    assert (call.tensor[..., 2].abs() > 0).any()


def test_body_com_offset_zero_when_ranges_missing() -> None:
    """Calling with empty ranges should still record a shape-correct zero tensor."""
    env = _FakeEnv(num_envs=2, link_names=["a"])
    dr.body_com_offset(env, env_ids=None, asset_cfg=_link_cfg(env, "a"), ranges=None)
    call = env.robot.com_shift
    assert call is not None
    assert call.tensor.shape == (2, 1, 3)
    assert torch.all(call.tensor == 0.0)


# --------------------------------------------------------------------- body_mass_offset


def test_body_mass_offset_samples_within_range() -> None:
    torch.manual_seed(3)
    env = _FakeEnv(num_envs=256, link_names=["base", "torso"])
    dr.body_mass_offset(
        env, env_ids=None, asset_cfg=SceneEntityCfg(name="robot"), ranges=(-0.5, 0.5)
    )
    call = env.robot.mass_shift
    assert call is not None
    assert call.tensor.shape == (256, 2)
    assert torch.all(call.tensor >= -0.5)
    assert torch.all(call.tensor <= 0.5)


# --------------------------------------------------------------------- encoder_bias


def test_encoder_bias_writes_uniform_samples_into_robot_state() -> None:
    torch.manual_seed(4)
    env = _FakeEnv(num_envs=32, joint_names=["hip", "knee", "ankle"])
    dr.encoder_bias(
        env,
        env_ids=None,
        asset_cfg=SceneEntityCfg(name="robot"),
        bias_range=(-0.01, 0.01),
    )
    assert env.robot_state is not None
    bias = env.robot_state.encoder_bias
    assert bias.shape == (32, 3)
    assert torch.all(bias >= -0.01) and torch.all(bias <= 0.01)
    # Distribution should span both signs (with 32×3=96 samples this is essentially certain).
    assert (bias > 0).any().item() and (bias < 0).any().item()


def test_encoder_bias_respects_env_ids_subset() -> None:
    env = _FakeEnv(num_envs=4, joint_names=["hip", "knee"])
    assert env.robot_state is not None
    env.robot_state.encoder_bias.fill_(0.0)
    dr.encoder_bias(
        env,
        env_ids=torch.tensor([1, 3]),
        asset_cfg=SceneEntityCfg(name="robot"),
        bias_range=(0.01, 0.01),
    )
    bias = env.robot_state.encoder_bias
    # Touched envs got the constant 0.01; untouched stay at 0.
    assert torch.allclose(bias[1], torch.tensor([0.01, 0.01]), atol=1e-6)
    assert torch.allclose(bias[3], torch.tensor([0.01, 0.01]), atol=1e-6)
    assert torch.all(bias[0] == 0.0)
    assert torch.all(bias[2] == 0.0)


def test_encoder_bias_filters_to_named_joints_only() -> None:
    env = _FakeEnv(num_envs=2, joint_names=["hip", "knee", "ankle"])
    assert env.robot_state is not None
    env.robot_state.encoder_bias.fill_(0.0)
    dr.encoder_bias(
        env,
        env_ids=None,
        asset_cfg=_joint_cfg(env, "knee"),
        bias_range=(0.02, 0.02),
    )
    bias = env.robot_state.encoder_bias
    # Only the knee column is filled; hip and ankle stay zero.
    assert torch.all(bias[:, 0] == 0.0)
    assert torch.allclose(bias[:, 1], torch.full((2,), 0.02), atol=1e-6)
    assert torch.all(bias[:, 2] == 0.0)


def test_joint_pos_rel_does_not_add_encoder_bias() -> None:
    """mjlab parity: ``joint_pos_rel`` returns the raw physical offset.

    Pre-parity behaviour added ``encoder_bias`` to the obs, which cancelled the
    matching action-side subtraction and silently neutralised the encoder-bias
    DR signal. With the fix the policy must learn to compensate.
    """
    env = _FakeEnv(num_envs=2, joint_names=["hip", "knee"])
    assert env.robot_state is not None
    env.robot_state.joint_pos = torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    env.default_joint_pos = torch.tensor([0.0, 0.0])
    env.robot_state.encoder_bias = torch.tensor([[0.01, -0.02], [0.03, 0.04]])
    obs = joint_pos_rel(env)
    expected = torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    assert torch.allclose(obs, expected, atol=1e-6)


def test_encoder_bias_subtracts_in_joint_position_action_target() -> None:
    """End-to-end: ``JointPositionAction.process_actions`` subtracts the bias.

    The physical joint sits ``-bias`` away from the policy's nominal command;
    ``joint_pos_rel`` surfaces that offset directly so the policy must learn
    to compensate — that's exactly the sim2real perturbation the encoder-bias
    DR models.
    """
    from genelab.mdp.actions.joint_position import (
        JointPositionAction,
        JointPositionActionCfg,
    )

    env = _FakeEnv(num_envs=2, joint_names=["hip", "knee"])
    assert env.robot_state is not None
    env.default_joint_pos = torch.tensor([0.5, -0.3])
    # Fixed bias so the test is deterministic.
    env.robot_state.encoder_bias = torch.tensor([[0.01, -0.02], [0.03, 0.04]])

    cfg = JointPositionActionCfg(scale=1.0, joint_names=(".*",))
    term = JointPositionAction(cfg, env)  # type: ignore[arg-type]
    action = torch.tensor([[0.1, 0.2], [-0.1, 0.0]])
    term.process_actions(action)

    # target = default + scale * action - bias
    expected = env.default_joint_pos.unsqueeze(0) + action - env.robot_state.encoder_bias
    assert torch.allclose(term._target, expected, atol=1e-6)
