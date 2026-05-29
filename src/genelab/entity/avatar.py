"""Avatar entity — visualization-only ghost / reference body.

Wraps Genesis 1.0's :class:`gs.materials.Kinematic`-backed kinematic entity
(the v0.4.1 "avatar" concept rebadged in v1.0): a body that participates in
the scene's visual / sensor pipeline but is **ignored** by physics — no
collision response, no constraint solving, no contact forces. Useful as a
scripted scene actor whose pose the env writes each step (motion clip
playback, target ghost arrow, reference robot in a tracking task) without
the actor exerting force on the rest of the simulation.

Sibling to :class:`~genelab.entity.RigidObject` — same two-phase lifecycle
(``spawn`` before ``scene.build``, no ``bind`` step) but adds a
``set_pos`` / ``set_quat`` write surface for the env's per-step pose
update.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class AvatarCfg:
    """Description of a kinematic (visualization-only) scene entity.

    The morph / size surface mirrors :class:`~genelab.entity.RigidObjectCfg` —
    same primitives plus ``"mesh"`` / ``"mjcf"`` for external assets. The
    material is fixed at ``gs.materials.Kinematic`` so the entity stays
    physics-inert regardless of the morph chosen.
    """

    morph: Literal["box", "sphere", "mesh", "mjcf"] = "box"
    file: str | None = None
    size: tuple[float, ...] = (0.1, 0.1, 0.1)
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


class Avatar:
    """Kinematic scene entity wrapping a Genesis ``Kinematic``-material handle.

    Avatars do not participate in physics; per-step pose updates flow through
    :meth:`set_pose`. The Genesis handle keeps the entity name visible to
    sensors / cameras but the rigid solver leaves it alone.
    """

    def __init__(self, cfg: AvatarCfg, *, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self._gs_handle: Any = None

    def spawn(self, gs_scene: Any) -> None:
        """Pre-build: build the morph and add the entity with a ``Kinematic`` material."""
        import genesis as gs  # type: ignore[import-not-found]

        morph_kind = self.cfg.morph
        if morph_kind == "box":
            if len(self.cfg.size) != 3:
                raise ValueError(f"Avatar(name={self.name!r}, morph='box') requires size=(x, y, z)")
            morph = gs.morphs.Box(
                size=tuple(self.cfg.size),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
            )
        elif morph_kind == "sphere":
            if len(self.cfg.size) != 1:
                raise ValueError(
                    f"Avatar(name={self.name!r}, morph='sphere') requires size=(radius,)"
                )
            morph = gs.morphs.Sphere(
                radius=float(self.cfg.size[0]),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
            )
        elif morph_kind in ("mesh", "mjcf"):
            if not self.cfg.file:
                raise ValueError(
                    f"Avatar(name={self.name!r}, morph={morph_kind!r}) requires cfg.file"
                )
            morph_cls = gs.morphs.Mesh if morph_kind == "mesh" else gs.morphs.MJCF
            morph = morph_cls(
                file=str(self.cfg.file),
                pos=tuple(self.cfg.init_pos),
                quat=tuple(self.cfg.init_quat),
            )
        else:
            raise ValueError(
                f"Avatar(name={self.name!r}): unknown morph {morph_kind!r}; "
                "expected one of 'box' / 'sphere' / 'mesh' / 'mjcf'"
            )

        self._gs_handle = gs_scene.add_entity(morph, material=gs.materials.Kinematic())

    def set_pose(
        self,
        pos: tuple[float, float, float] | Any,
        quat: tuple[float, float, float, float] | Any,
        *,
        envs_idx: Any = None,
    ) -> None:
        """Write the entity's world-frame pose for the next render step.

        ``pos`` and ``quat`` follow the Genesis batched shape conventions —
        ``(B, 3)`` and ``(B, 4)`` tensors (or 1-D iterables for the single-env
        case). ``envs_idx`` slices a subset of envs; ``None`` writes every env.
        Avatars have no joint state, so ``set_pose`` is the entire write API.
        """
        if self._gs_handle is None:
            raise RuntimeError(f"Avatar(name={self.name!r}).set_pose called before spawn")
        self._gs_handle.set_pos(pos, envs_idx=envs_idx)
        self._gs_handle.set_quat(quat, envs_idx=envs_idx)

    @property
    def gs_handle(self) -> Any:
        """The raw Genesis kinematic-entity handle (``None`` before ``spawn``)."""
        return self._gs_handle
