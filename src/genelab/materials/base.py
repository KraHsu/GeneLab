"""Base class for Genesis material wrappers.

Each material ``*Cfg`` is a flat dataclass whose fields mirror a ``gs.materials``
constructor one-for-one. A field left at ``None`` means "use the Genesis default":
:meth:`MaterialCfg.build` forwards only the fields the caller set, so an untouched
cfg produces a Genesis material identical to calling the constructor with no
arguments.

The module never imports ``genesis`` at load time; :meth:`build` receives the ``gs``
module as an argument (mirrors :meth:`genelab.entity.RigidObject.spawn`), keeping
``import genelab.configs`` free of torch.
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar


@dataclass
class MaterialCfg:
    """Base for declarative Genesis material configs.

    Subclasses set :attr:`_gs_path` to the constructor's dotted location under
    ``gs.materials`` (``"Rigid"``, ``"MPM.Elastic"``, …) and :attr:`gs_solver` to the
    solver family they belong to (``"rigid"``, ``"mpm"``, ``"fem"``, ``"pbd"``,
    ``"sph"``, ``"sf"``, ``"tool"``, ``"kinematic"``). The scene reads
    :meth:`required_solvers` to decide which solver ``*_options`` to enable on
    ``gs.Scene`` (see :class:`~genelab.scene.InteractiveScene`).
    """

    gs_solver: ClassVar[str] = "rigid"
    _gs_path: ClassVar[str] = ""

    def required_solvers(self) -> set[str]:
        """Solver families this material needs enabled on the scene."""
        return {type(self).gs_solver}

    def _kwargs(self) -> dict[str, Any]:
        """Set fields (non-``None``) keyed by their Genesis constructor names.

        Field names mirror Genesis exactly, so the default is the dataclass fields
        verbatim with ``None`` dropped. This is the seam tests assert against without
        importing Genesis.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out

    def _factory(self, gs: Any) -> Any:
        obj: Any = gs.materials
        for part in type(self)._gs_path.split("."):
            obj = getattr(obj, part)
        return obj

    def build(self, gs: Any) -> Any:
        """Construct the backing ``gs.materials`` object from the set fields."""
        return self._factory(gs)(**self._kwargs())
