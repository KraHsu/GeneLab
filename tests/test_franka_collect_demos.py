"""Regression test for the Franka demo-collection FSM + collect() pipeline.

Guards against the vNext ``env.robot_state.*`` removal (ADR-0019): the demo FSM must
read EE/joint state through ``env.articulations["robot"].data.*``. Drives the REAL
``FrankaPickAndPlaceFsm`` and ``collect()`` against a lightweight fake env (current
Articulation API) — no Genesis runtime — then round-trips the ``.npz`` and asserts the
HER-prefill keys and shapes. Before the fix this raised
``AttributeError: 'ManagerBasedRlEnv' object has no attribute 'robot_state'``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from genelab_franka.constants import (  # noqa: E402
    CUBE_ENTITY_NAME,
    EE_LINK,
)

_POLICY_DIM = 35
_GOAL_DIM = 3
_ACTION_DIM = 4
_GOAL_ATTR = "_franka_pp_goal_w"
# Franka: 7 arm joints then 2 finger joints (the FSM picks joints containing
# "finger_joint" and reads ``data.joint_pos``).
_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
_LINK_NAMES = ["link0", EE_LINK, "left_finger", "right_finger"]


class _FakeData:
    def __init__(self, num_envs: int) -> None:
        self.link_pos = torch.zeros(num_envs, len(_LINK_NAMES), 3)
        self.joint_pos = torch.zeros(num_envs, len(_JOINT_NAMES))


class _FakeArticulation:
    def __init__(self, num_envs: int) -> None:
        self.link_names = list(_LINK_NAMES)
        self.joint_names = list(_JOINT_NAMES)
        self.data = _FakeData(num_envs)


class _FakeHandle:
    def __init__(self, num_envs: int) -> None:
        self._pos = torch.zeros(num_envs, 3)

    def get_pos(self) -> torch.Tensor:
        return self._pos


class _FakeScene:
    def __init__(self, num_envs: int) -> None:
        class _Cube:
            gs_handle = _FakeHandle(num_envs)

        self.rigid_objects = {CUBE_ENTITY_NAME: _Cube()}


class _FakeFrankaEnv:
    """Minimal env satisfying the FSM + collect() contract (no Genesis)."""

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _FakeScene(num_envs)
        self.articulations = {"robot": _FakeArticulation(num_envs)}
        setattr(self, _GOAL_ATTR, torch.zeros(num_envs, _GOAL_DIM))

    def _obs(self) -> dict[str, Any]:
        return {
            "policy": torch.zeros(self.num_envs, _POLICY_DIM),
            "achieved_goal": torch.zeros(self.num_envs, _GOAL_DIM),
            "desired_goal": torch.zeros(self.num_envs, _GOAL_DIM),
        }

    def reset(self) -> Any:
        return self._obs(), {}

    def step(self, _action: Any) -> Any:
        false = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), false, false, {}


def test_demo_fsm_reads_state_through_articulation_data() -> None:
    """The FSM accessors must use ``articulations[...].data.*`` (no robot_state)."""
    from genelab_franka.demo_fsm import FrankaPickAndPlaceFsm

    env = _FakeFrankaEnv(num_envs=4)
    fsm = FrankaPickAndPlaceFsm(env)

    assert fsm._ee_pos().shape == (4, 3)
    assert fsm._gripper_width().shape == (4,)
    assert fsm.step().shape == (4, _ACTION_DIM)


def test_collect_writes_npz_with_expected_keys_and_shapes(tmp_path: Path) -> None:
    from genelab_franka.collect_demos import collect, save_demos
    from genelab_franka.demo_fsm import FrankaPickAndPlaceFsm

    num_envs, steps = 4, 3
    env = _FakeFrankaEnv(num_envs=num_envs)
    out = collect(env, FrankaPickAndPlaceFsm(env), steps=steps)

    path = tmp_path / "demos.npz"
    save_demos(out, path)
    assert path.exists()

    loaded = np.load(path)
    expected_shapes = {
        "obs_observation": (steps, num_envs, _POLICY_DIM),
        "obs_achieved_goal": (steps, num_envs, _GOAL_DIM),
        "obs_desired_goal": (steps, num_envs, _GOAL_DIM),
        "next_obs_observation": (steps, num_envs, _POLICY_DIM),
        "next_obs_achieved_goal": (steps, num_envs, _GOAL_DIM),
        "next_obs_desired_goal": (steps, num_envs, _GOAL_DIM),
        "actions": (steps, num_envs, _ACTION_DIM),
        "rewards": (steps, num_envs),
        "dones": (steps, num_envs),
        "truncateds": (steps, num_envs),
    }
    assert set(loaded.files) == set(expected_shapes)
    for key, shape in expected_shapes.items():
        assert loaded[key].shape == shape, f"{key}: {loaded[key].shape} != {shape}"
    assert loaded["dones"].dtype == np.bool_
    assert loaded["actions"].dtype == np.float32
