"""Recording module tests.

Pure-Python unit tests cover the data-source resolver (no Genesis needed). The two
integration tests use the existing ``genesis_runtime`` fixture (skipped on hosts
without ``libEGL``) to drive a real env with NPZ + ``save_on_reset`` recorders.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from genelab.recording import (
    NPZFileCfg,
    PyQtPlotCfg,
    RecorderBridge,
    RecordingCfg,
    VideoFileCfg,
)
from genelab.recording.resolve import (
    is_sensor_source,
    resolve_data_func,
    validate_output_compatibility,
)


# ---------------------------------------------------------------------------- fakes


@dataclass
class _FakeIMUData:
    orientation: torch.Tensor
    lin_acc_b: torch.Tensor


class _FakeSensor:
    def __init__(self, data: Any) -> None:
        self._data = data

    @property
    def data(self) -> Any:
        return self._data


class _FakeCameraData:
    def __init__(self, rgb: torch.Tensor) -> None:
        self.rgb = rgb


class _FakeCameraSensor:
    """Resolver inspects ``type(sensor).__name__ == 'CameraSensor'`` — match it."""

    def __init__(self, data: _FakeCameraData) -> None:
        self._data = data

    @property
    def data(self) -> _FakeCameraData:
        return self._data


# Rename so the resolver's ``type(sensor).__name__`` check fires.
_FakeCameraSensor.__name__ = "CameraSensor"


class _FakeEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.foo: torch.Tensor = torch.tensor([1.0, 2.0])


def _bridge_with_sensors(**sensors: Any) -> RecorderBridge:
    bridge = RecorderBridge.__new__(RecorderBridge)
    bridge.scene = None  # type: ignore[assignment]
    bridge.sensors = dict(sensors)
    bridge.entities = {}
    bridge.env = _FakeEnv(num_envs=2)
    bridge.handles = []
    bridge._last_reset_step = -1
    return bridge


# ---------------------------------------------------------------------------- resolver tests


def test_resolve_sensor_field_returns_tensor() -> None:
    imu = _FakeSensor(
        _FakeIMUData(
            orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            lin_acc_b=torch.zeros(2, 3),
        )
    )
    bridge = _bridge_with_sensors(imu=imu)
    rec_cfg = RecordingCfg(
        name="r",
        source="imu",
        field="orientation",
        env_idx=0,
        outputs=(NPZFileCfg(filename="x.npz"),),
    )
    func = resolve_data_func(rec_cfg, bridge)
    out = func()
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4,)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_resolve_callable_arity_one() -> None:
    bridge = _bridge_with_sensors()

    def src(env: _FakeEnv) -> torch.Tensor:
        return env.foo

    rec_cfg = RecordingCfg(
        name="r", source=src, env_idx=None, outputs=(NPZFileCfg(filename="x.npz"),)
    )
    func = resolve_data_func(rec_cfg, bridge)
    out = func()
    assert torch.equal(out, torch.tensor([1.0, 2.0]))


def test_resolve_callable_arity_zero() -> None:
    bridge = _bridge_with_sensors()

    def src() -> float:
        return 3.14

    rec_cfg = RecordingCfg(
        name="r", source=src, env_idx=None, outputs=(NPZFileCfg(filename="x.npz"),)
    )
    func = resolve_data_func(rec_cfg, bridge)
    assert func() == pytest.approx(3.14)


def test_resolve_rejects_dataclass_payload_without_field() -> None:
    imu = _FakeSensor(
        _FakeIMUData(
            orientation=torch.zeros(2, 4),
            lin_acc_b=torch.zeros(2, 3),
        )
    )
    bridge = _bridge_with_sensors(imu=imu)
    rec_cfg = RecordingCfg(
        name="r", source="imu", field=None, outputs=(NPZFileCfg(filename="x.npz"),)
    )
    func = resolve_data_func(rec_cfg, bridge)
    with pytest.raises(ValueError, match="specify field"):
        func()


def test_resolve_rejects_callable_source_with_field() -> None:
    bridge = _bridge_with_sensors()

    def src() -> float:
        return 0.0

    with pytest.raises(ValueError, match="field=... is only valid"):
        resolve_data_func(
            RecordingCfg(
                name="r",
                source=src,
                field="orientation",
                outputs=(NPZFileCfg(filename="x.npz"),),
            ),
            bridge,
        )


def test_resolve_camera_video_indexes_env() -> None:
    rgb = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 4, 3)
    cam = _FakeCameraSensor(_FakeCameraData(rgb=rgb))
    bridge = _bridge_with_sensors(cam=cam)
    rec_cfg = RecordingCfg(
        name="r",
        source="cam",
        outputs=(VideoFileCfg(filename="out.mp4", env_idx=1),),
    )
    func = resolve_data_func(rec_cfg, bridge)
    frame = func()
    assert isinstance(frame, torch.Tensor)
    assert frame.shape == (4, 4, 3)
    # env_idx=1 should give the second slice.
    assert torch.equal(frame, rgb[1])


def test_validate_rejects_video_with_non_camera_source() -> None:
    imu = _FakeSensor(_FakeIMUData(orientation=torch.zeros(2, 4), lin_acc_b=torch.zeros(2, 3)))
    bridge = _bridge_with_sensors(imu=imu)
    with pytest.raises(ValueError, match="CameraSensor"):
        validate_output_compatibility(
            RecordingCfg(
                name="r",
                source="imu",
                field="orientation",
                outputs=(VideoFileCfg(filename="x.mp4"),),
            ),
            bridge,
        )


def test_validate_rejects_plot_with_camera_source() -> None:
    cam = _FakeCameraSensor(_FakeCameraData(rgb=torch.zeros(2, 4, 4, 3, dtype=torch.uint8)))
    bridge = _bridge_with_sensors(cam=cam)
    with pytest.raises(ValueError, match="plot outputs"):
        validate_output_compatibility(
            RecordingCfg(
                name="r",
                source="cam",
                outputs=(PyQtPlotCfg(title="bad"),),
            ),
            bridge,
        )


def test_is_sensor_source_predicate() -> None:
    assert is_sensor_source(RecordingCfg(name="r", source="imu")) is True
    assert is_sensor_source(RecordingCfg(name="r", source=lambda: 0.0)) is False


def test_env_idx_none_passes_through_full_batch() -> None:
    imu = _FakeSensor(
        _FakeIMUData(
            orientation=torch.eye(4).unsqueeze(0).expand(2, 4, 4),  # shape (2, 4, 4)
            lin_acc_b=torch.zeros(2, 3),
        )
    )
    bridge = _bridge_with_sensors(imu=imu)
    rec_cfg = RecordingCfg(
        name="r",
        source="imu",
        field="lin_acc_b",
        env_idx=None,
        outputs=(NPZFileCfg(filename="x.npz"),),
    )
    func = resolve_data_func(rec_cfg, bridge)
    out = func()
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3)  # full batch preserved


# ---------------------------------------------------------------------------- integration tests


CART_LINK = "cart"
POLE_LINK = "pole"


def _build_recording_env(tmp_path: Path, save_on_reset: bool = False) -> Any:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.sensor import BodyVelocitySensorCfg, IMUSensorCfg
    from genelab_inverted_pendulum.single.env_cfg import inverted_pendulum_env_cfg

    cfg = inverted_pendulum_env_cfg(play=True)
    cfg.simulation.vis = False
    cfg.simulation.gpu = False
    cfg.device = "cpu"
    cfg.scene.sensors = (
        BodyVelocitySensorCfg(name="pole_ang_vel", link_name=POLE_LINK, measure="ang_vel"),
        IMUSensorCfg(name="cart_imu", link_name=CART_LINK, gravity_bias=True),
    )
    cfg.scene.recordings = (
        RecordingCfg(
            name="cart_lin_acc",
            source="cart_imu",
            field="lin_acc_b",
            outputs=(
                NPZFileCfg(
                    filename=str(tmp_path / "cart_lin_acc.npz"),
                    save_on_reset=save_on_reset,
                ),
            ),
        ),
        RecordingCfg(
            name="pole_pos",
            source=lambda env: env.robot_state.joint_pos[:, 0],
            outputs=(NPZFileCfg(filename=str(tmp_path / "pole_pos.npz")),),
        ),
    )
    return ManagerBasedRlEnv(cfg)


def test_recording_npz_smoke(genesis_runtime: Any, tmp_path: Path) -> None:
    del genesis_runtime
    env = _build_recording_env(tmp_path)
    try:
        env.reset()
        zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        for _ in range(8):
            env.step(zero)
    finally:
        env.close()

    npz_path = tmp_path / "cart_lin_acc.npz"
    assert npz_path.exists(), f"expected {npz_path} to exist"
    loaded = np.load(npz_path)
    assert "timestamp" in loaded.files
    assert "data" in loaded.files
    assert len(loaded["timestamp"]) >= 1


def test_recording_default_hz_matches_control_rate_for_sensor_source(
    genesis_runtime: Any, tmp_path: Path
) -> None:
    del genesis_runtime
    env = _build_recording_env(tmp_path)
    try:
        bridge = env.scene.recorder_bridge
        assert bridge is not None
        decimation = env.cfg.decimation

        sensor_handles = [h for h in bridge.handles if h.is_sensor_source]
        callable_handles = [h for h in bridge.handles if not h.is_sensor_source]
        assert sensor_handles, "expected at least one sensor-source recorder"
        assert callable_handles, "expected at least one callable-source recorder"

        for handle in sensor_handles:
            assert handle.recorder._steps_per_sample == decimation
        for handle in callable_handles:
            # Callable sources keep ``hz=None`` (every physics tick).
            assert handle.recorder._steps_per_sample == 1
    finally:
        env.close()
