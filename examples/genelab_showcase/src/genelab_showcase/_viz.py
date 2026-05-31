"""Lazy pyqtgraph window helper shared by the showcase runners.

Every showcase that renders a live Qt window (point cloud, camera image, plot) opens it lazily
on the first viz tick and tolerates a missing pyqtgraph / display: the window is built once via
a builder callback, and if that fails the runner silently continues without it (never retrying).
"""

from collections.abc import Callable
from typing import Any


class LazyQtWindows:
    """One-time pyqtgraph app + window setup with a best-effort fallback.

    Usage::

        self._qt = LazyQtWindows()
        ...
        if self._qt.ensure(self._build, label="my showcase"):
            self._item.setData(...)
            self._qt.process()

    :meth:`ensure` calls ``builder(app)`` exactly once (``app`` is the ``QApplication``). It
    returns ``True`` when pyqtgraph is importable and the build succeeds, ``False`` otherwise —
    and caches the verdict so a failed import is not retried on every step.
    """

    def __init__(self) -> None:
        self._app: Any = None
        self._tried = False
        self._ready = False

    def ensure(self, builder: Callable[[Any], None], *, label: str) -> bool:
        if self._tried:
            return self._ready
        self._tried = True
        try:
            import pyqtgraph as pg

            self._app = pg.mkQApp(label)
            builder(self._app)
            self._ready = True
        except Exception as exc:  # noqa: BLE001 - viz is best-effort, never break the sim
            print(f"{label}: live window unavailable ({exc})")
        return self._ready

    def process(self) -> None:
        """Pump the Qt event loop so the window stays responsive."""
        if self._app is not None:
            self._app.processEvents()
