"""ImGui panel cookbook: how to add common GUI widgets to the Genesis viewer.

GeneLab does not reimplement a GUI toolkit — it forwards plain ``callback(imgui)``
functions to Genesis's built-in ImGui overlay via :attr:`SimulationCfg.panels`. Adding a
GUI element is therefore exactly as light as in raw Genesis: write one function that calls
``imgui.slider_float(...)`` / ``imgui.checkbox(...)`` / etc. and append it to ``panels``.

* :mod:`~genelab_examples.gui_panels.widgets` — a copy-paste recipe per widget family
  (text, buttons, sliders, inputs, toggles, dropdowns, color, layout).
* :mod:`~genelab_examples.gui_panels.config` — an env cfg whose ``simulation.panels`` is
  preloaded with the whole cookbook.
* :mod:`~genelab_examples.gui_panels.env` — a runnable standalone scene that forwards those
  panels to the viewer with a single :func:`genelab.scene.register_viewer_panels` call.

Run it::

    genelab play GeneLab-GUI-Panels-Demo-v0 --vis

Needs the optional ImGui overlay dependency: ``uv sync --extra imgui``.
"""
