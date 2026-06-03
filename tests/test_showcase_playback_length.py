"""ShowcaseRunner playback-length gating: soft ``--steps`` vs hard ``--max-steps``.

Mirrors the ``play_task`` viewer-gate test in ``tests/test_rl_pipeline.py`` for the
non-RL scripted showcase runner, so ``genelab play`` behaves identically whichever
runner backs the task:

* headless (``vis=False``) caps at ``simulation.steps`` (the soft ``--steps`` config),
* a viewer (``vis=True``) runs unbounded until the window closes (soft steps ignored),
* an explicit ``max_steps`` (``--max-steps``) is a hard cap that wins in either mode.

The env is faked (no Genesis) by monkeypatching ``ManagerBasedRlEnv`` in the runner
module, so the test asserts purely on how many steps the loop drives.
"""

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab_showcase import runner as showcase_runner  # noqa: E402


class _FakeEnv:
    """Minimal stand-in for ``ManagerBasedRlEnv``: counts steps, can self-close."""

    def __init__(self, *, close_at: int | None = None) -> None:
        self.num_envs = 1
        self.num_actions = 2
        self.device = "cpu"
        self.viewer_closed = False
        self.steps_taken = 0
        self._close_at = close_at

    def reset(self) -> None:
        return None

    def step(self, _action: Any) -> None:
        self.steps_taken += 1
        if self._close_at is not None and self.steps_taken >= self._close_at:
            self.viewer_closed = True

    def close(self) -> None:
        return None


def _runner(monkeypatch: pytest.MonkeyPatch, *, vis: bool, steps: int, fake: _FakeEnv):
    monkeypatch.setattr(showcase_runner, "ensure_project_cache", lambda: None)
    monkeypatch.setattr(showcase_runner, "ManagerBasedRlEnv", lambda _cfg: fake)
    env_cfg = SimpleNamespace(simulation=SimpleNamespace(vis=vis, steps=steps, dt=0.01))
    # realtime=False disables wall-clock pacing so the loop runs at full speed.
    return showcase_runner.ShowcaseRunner(env_cfg, realtime=False)  # type: ignore[arg-type]


def test_headless_caps_at_simulation_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeEnv()
    _runner(monkeypatch, vis=False, steps=5, fake=fake).play()
    assert fake.steps_taken == 5


def test_viewer_runs_until_window_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    # vis + no max_steps => unbounded; the soft steps=3 is ignored and only the
    # viewer-close (here after 6 steps) ends the loop.
    fake = _FakeEnv(close_at=6)
    _runner(monkeypatch, vis=True, steps=3, fake=fake).play()
    assert fake.steps_taken == 6


def test_max_steps_is_hard_cap_with_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    # vis + max_steps => stops after exactly max_steps even though the window is open
    # and the soft steps is far larger.
    fake = _FakeEnv()
    _runner(monkeypatch, vis=True, steps=200, fake=fake).play(max_steps=4)
    assert fake.steps_taken == 4


def test_max_steps_overrides_headless_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeEnv()
    _runner(monkeypatch, vis=False, steps=100, fake=fake).play(max_steps=2)
    assert fake.steps_taken == 2
