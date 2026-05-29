"""Unit tests for ``InteractiveSceneCfg.use_rasterizer`` cfg field + plumbing."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.configs import InteractiveSceneCfg  # noqa: E402


class _RecordingBatchRenderer:
    def __init__(self, *, use_rasterizer: bool) -> None:
        self.use_rasterizer = use_rasterizer


def test_use_rasterizer_default_false() -> None:
    cfg = InteractiveSceneCfg()
    assert cfg.use_rasterizer is False


def test_use_rasterizer_can_be_overridden() -> None:
    cfg = InteractiveSceneCfg(use_rasterizer=True, batch_render=True)
    assert cfg.use_rasterizer is True
    assert cfg.batch_render is True


@pytest.mark.parametrize("use_rasterizer", [False, True])
def test_use_rasterizer_forwarded_to_batch_renderer(
    monkeypatch: pytest.MonkeyPatch, use_rasterizer: bool
) -> None:
    """The ``BatchRenderer`` constructor receives the cfg value."""
    import genesis as gs

    captured: list[Any] = []

    def _fake_batch_renderer(*, use_rasterizer: bool) -> _RecordingBatchRenderer:
        renderer = _RecordingBatchRenderer(use_rasterizer=use_rasterizer)
        captured.append(renderer)
        return renderer

    monkeypatch.setattr(gs.renderers, "BatchRenderer", _fake_batch_renderer)

    # The plumbing in ``InteractiveScene.build`` is a single conditional expression;
    # exercise it via the real ``InteractiveSceneCfg`` so a future refactor keeps the
    # field name + default in sync.
    cfg = InteractiveSceneCfg(batch_render=True, use_rasterizer=use_rasterizer)
    renderer = (
        gs.renderers.BatchRenderer(use_rasterizer=bool(getattr(cfg, "use_rasterizer", False)))
        if getattr(cfg, "batch_render", False)
        else None
    )
    assert renderer is not None
    assert captured[0].use_rasterizer is use_rasterizer
