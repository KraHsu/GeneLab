"""Actuator-showcase runner: live joint-1 tracking curves + interactive PD / sine sliders.

Drives joint 1 with a sine position target through one or more arm actuators and renders the
commanded target plus each arm's achieved position and tracking error as live pyqtgraph curves.

* The base showcase drives a single IdealPD arm.
* The MlpResidual variant drives *two* arms from the same scripted target — ``robot`` (with the
  learned residual) and ``robot_ideal`` (plain IdealPD) — so the two curves overlay "with
  residual" vs "without residual".

Four sliders retune the demo live: ``kp`` / ``kv`` write straight into every arm's PD gains
(under-damped jitter vs over-damped lag), and ``amplitude`` / ``period`` reshape the sine.
Nothing is written to disk.
"""

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from genelab_showcase._viz import LazyQtWindows
from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

_HISTORY = 400  # samples kept in the rolling plot window
_ACTION_SCALE = 0.5  # matches the actuator's action_scale in env_cfg (target = default + scale*raw)
_ARM_GROUP = "panda_arm"  # the arm actuator group name within each Franka
_TARGET_RGB = (255, 200, 40)
# Live-slider spec: (attr, label, slider min, slider max, value-per-tick, format).
_SLIDERS: tuple[tuple[str, str, int, int, float, str], ...] = (
    ("_kp", "kp (stiffness)", 0, 800, 1.0, "{:.0f}"),
    ("_kv", "kv (damping)", 0, 160, 1.0, "{:.0f}"),
    ("_amp", "sine amplitude (rad)", 0, 120, 0.01, "{:.2f}"),
    ("_period", "sine period (steps)", 40, 400, 1.0, "{:.0f}"),
)


class _Arm:
    """Per-arm live state: which action slot to drive, which robot to read, and its curves."""

    def __init__(self, label: str, term_key: str, robot_name: str, rgb: tuple[int, int, int]):
        self.label = label
        self.term_key = term_key
        self.robot_name = robot_name
        self.rgb = rgb
        self.slot: int = -1  # joint-1 action slot (resolved on first step)
        self.j1: int = 0  # joint-1 index in this robot's joint order (resolved by name)
        self.actuator: Any = None
        self.actual: deque[float] = deque(maxlen=_HISTORY)
        self.error: deque[float] = deque(maxlen=_HISTORY)
        self.curve_actual: Any = None
        self.curve_error: Any = None


class ActuatorShowcaseRunner(ShowcaseRunner):
    """Sine joint-1 target through an IdealPD arm; live tracking curves + PD / sine sliders."""

    task_slug = "actuators"
    # (curve label, action-term key, robot name, RGB) — one entry per arm driven/plotted.
    arms_spec: tuple[tuple[str, str, str, tuple[int, int, int]], ...] = (
        ("actual", "panda_arm", "robot", (40, 200, 255)),
    )

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg)
        self._arms = [_Arm(*spec) for spec in self.arms_spec]
        self._ready = False
        self._default = 0.0  # joint-1 default position (shared; both Frankas are identical)
        # Live slider state; defaults match env_cfg's actuator + the scripted sine.
        self._kp = 400.0
        self._kv = 40.0
        self._amp = 0.8  # raw sine amplitude in rad (before _ACTION_SCALE)
        self._period = 200.0  # control steps per sine period
        self._phase = 0.0  # accumulated so a mid-run period change does not jump the sine
        self._target = 0.0  # commanded joint-1 target this step (shared by action + plot)
        self._robot_names: frozenset[str] = frozenset()  # distinct robots to refresh (resolved)
        self._t: deque[int] = deque(maxlen=_HISTORY)
        self._cmd: deque[float] = deque(maxlen=_HISTORY)  # commanded-target history (shared)
        self._qt = LazyQtWindows()
        self._win: Any = None
        self._c_target: Any = None

    def _resolve(self, env: "ManagerBasedRlEnv") -> None:
        """Resolve each arm's joint-1 action slot + actuator handle (once, on the first step)."""
        offsets: dict[str, int] = {}
        offset = 0
        # No public accessor for the action-term registry; reach it through the manager.
        terms: dict[str, Any] = getattr(env.action_manager, "_terms")
        for key, term in terms.items():
            offsets[key] = offset
            offset += term.action_dim
        for arm in self._arms:
            # joint 1 is the first joint of an arm term, so its slot is the term's base offset.
            arm.slot = offsets[arm.term_key]
            robot = env.articulations[arm.robot_name]
            arm.j1 = robot.joint_names.index("joint1")
            # No public accessor for the actuator registry; reach the arm group through it.
            arm.actuator = getattr(robot, "_actuators")[_ARM_GROUP]
        first = self._arms[0]
        self._default = float(env.articulations[first.robot_name].default_joint_pos[first.j1])
        self._robot_names = frozenset(arm.robot_name for arm in self._arms)
        self._ready = True

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        if not self._ready:
            self._resolve(env)
        # Advance the sine by accumulated phase so amplitude / period sliders stay smooth.
        self._phase += 2.0 * math.pi / self._period
        raw = math.sin(self._phase) * self._amp
        self._target = self._default + _ACTION_SCALE * raw
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for arm in self._arms:
            # Push the live PD gains into each arm (its compute() reads these each sub-step).
            arm.actuator.stiffness[:] = self._kp
            arm.actuator.damping[:] = self._kv
            action[:, arm.slot] = raw
        return action

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if not self._ready:
            return
        # env.step only refreshes the primary articulation's data cache, so pull fresh state for
        # every robot we read (the second arm is non-primary and would otherwise read stale zeros).
        for name in self._robot_names:
            env.articulations[name].refresh()
        self._t.append(step)
        self._cmd.append(self._target)
        for arm in self._arms:
            actual = float(env.articulations[arm.robot_name].data.joint_pos[0, arm.j1])
            arm.actual.append(actual)
            arm.error.append(self._target - actual)
        if not self._qt.ensure(self._build_window, label="actuator tracking"):
            return
        xs = list(self._t)
        self._c_target.setData(xs, list(self._cmd))
        for arm in self._arms:
            arm.curve_actual.setData(xs, list(arm.actual))
            arm.curve_error.setData(xs, list(arm.error))
        self._qt.process()

    def _build_window(self, app: Any) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        root = QtWidgets.QWidget()
        root.setWindowTitle("actuator tracking — joint 1 (PD / sine sliders)")
        root.resize(820, 620)
        vbox = QtWidgets.QVBoxLayout(root)

        glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(glw, stretch=1)
        p_track = glw.addPlot(row=0, col=0, title="joint 1 position (rad)")
        p_track.addLegend(offset=(10, 10))
        p_track.setLabel("bottom", "control step")
        self._c_target = p_track.plot(pen=pg.mkPen(_TARGET_RGB, width=2), name="target")
        p_err = glw.addPlot(row=1, col=0, title="tracking error = target − actual (rad)")
        p_err.addLegend(offset=(10, 10))
        p_err.setLabel("bottom", "control step")
        p_err.addLine(y=0.0, pen=pg.mkPen((120, 120, 120), width=1))
        for arm in self._arms:
            arm.curve_actual = p_track.plot(pen=pg.mkPen(arm.rgb, width=2), name=arm.label)
            arm.curve_error = p_err.plot(pen=pg.mkPen(arm.rgb, width=2), name=arm.label)

        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)
        for row, (attr, label, lo, hi, step_val, fmt) in enumerate(_SLIDERS):
            current = getattr(self, attr)
            tag = QtWidgets.QLabel(f"{label}: {fmt.format(current)}")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(lo)
            slider.setMaximum(hi)
            slider.setValue(round(current / step_val))

            def on_change(
                ticks: int, attr: str = attr, label: str = label, step_val: float = step_val,
                fmt: str = fmt, tag: QtWidgets.QLabel = tag,
            ) -> None:
                value = ticks * step_val
                setattr(self, attr, value)
                tag.setText(f"{label}: {fmt.format(value)}")

            slider.valueChanged.connect(on_change)
            grid.addWidget(tag, row, 0)
            grid.addWidget(slider, row, 1)

        root.show()
        self._win = root


class MlpResidualActuatorShowcaseRunner(ActuatorShowcaseRunner):
    """Two arms from one scripted target: ``robot`` (MlpResidual) vs ``robot_ideal`` (IdealPD).

    The two joint-1 responses overlay so the learned residual's effect is visible directly. The
    kp / kv sliders retune both arms' base PD gains together."""

    task_slug = "mlp_residual_actuator"
    arms_spec = (
        ("with residual", "resid_arm", "robot", (40, 200, 255)),
        ("without residual", "ideal_arm", "robot_ideal", (255, 120, 220)),
    )
