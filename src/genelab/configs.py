"""Core configuration objects and CLI override helpers."""

from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin, get_type_hints

# Heavy types are quoted so ``import genelab.configs`` does not transitively drag
# torch in through entity / sensor / terrains. Resolution happens lazily inside
# ``_field_annotation`` via ``get_type_hints``; that call is guarded against the
# NameError that fires when the heavy modules have not been imported yet.
# ``genelab.materials`` is a pure-dataclass module (no torch / genesis at import),
# so importing the solver-options group here is cheap and does not undo the lazy-import
# discipline the rest of this module follows for the heavy entity / sensor packages.
from genelab.materials.options import SolverOptionsCfg

if TYPE_CHECKING:
    from genelab.entity import ArticulationCfg, RigidObjectCfg
    from genelab.recording import RecordingCfg
    from genelab.sensor import SensorCfg
    from genelab.terrains import TerrainGeneratorCfg

type _Annotation = object


@dataclass
class SimulationCfg:
    """Genesis runtime / sim-loop settings (decoupled from scene composition)."""

    vis: bool = False
    # Genesis backend selector. ``None`` (default) follows the env's ``device``
    # (``cuda*`` -> GPU, else CPU) so a ``device="cuda"`` env runs the sim on the GPU
    # without a separate opt-in — avoiding the footgun where an unset ``gpu`` left the
    # whole sim on CPU (GPU idle, ~50x slower). ``True`` / ``False`` force the backend.
    gpu: bool | None = None
    steps: int = 240
    dt: float = 0.01
    substeps: int = 4
    num_envs: int = 1
    # Viewer FPS cap, decoupled from the physics rate (``1/dt``). Forwarded to
    # ``gs.options.ViewerOptions(max_FPS=...)`` when ``vis=True``. ``None`` runs the
    # viewer uncapped. ``ManagerBasedRlEnv`` only refreshes the viewer on the last tick
    # of its decimation loop, so the effective render rate is
    # ``min(render_fps, 1/(dt*decimation))``; if the control rate exceeds ``render_fps``,
    # the viewer's rate-limit will throttle the env step itself — lower ``render_fps`` or
    # raise ``decimation`` if that's not what you want.
    render_fps: int | None = 60
    # Initial viewer camera framing, forwarded to ``gs.options.ViewerOptions(camera_pos=...,
    # camera_lookat=...)`` when ``vis=True``. ``camera_lookat`` is also the trackball pivot,
    # so pointing it at the subject (e.g. a single robot's spawn) makes the mouse-wheel zoom
    # close in on that subject instead of the scene centroid. ``None`` leaves Genesis' own
    # default framing untouched.
    camera_pos: tuple[float, float, float] | None = None
    camera_lookat: tuple[float, float, float] | None = None
    # Blender-style free-fly navigation in the viewer: hold the right mouse button to look
    # around with the mouse and fly with WASD (Space up / Shift down); release to return to the
    # default orbit controls. A core viewer capability, on by default; ``move_speed`` is in m/s.
    fly_camera: bool = True
    fly_camera_speed: float = 3.0
    # ImGui overlay toggle for the Genesis viewer (v1.0+). Forwarded to
    # ``gs.options.ViewerOptions(enable_gui=...)`` when ``vis=True``. Off by default so
    # headless / scripted runs don't allocate the overlay. Setting ``panels`` (below)
    # turns the overlay on automatically, so this flag only matters when you want the
    # overlay's built-in controls without registering any custom panel.
    viewer_imgui: bool = False

    # Custom ImGui panels drawn in the Genesis viewer's overlay. Each entry is a
    # ``callback(imgui)`` invoked once per frame on the viewer thread with the live
    # ``imgui`` module (``imgui_bundle``); inside it you call ``imgui.slider_float(...)``,
    # ``imgui.checkbox(...)`` etc. exactly as you would in raw Genesis — GeneLab only
    # forwards each callback to the overlay's ``register_panel`` after ``scene.build``
    # (see :class:`~genelab.scene.InteractiveScene`). Adding a GUI element is therefore
    # no heavier than in Genesis: append one function here. A non-empty list implies the
    # ImGui overlay (no need to also set ``viewer_imgui=True``). Requires the ``imgui``
    # extra (``uv sync --extra imgui``); ignored entirely when ``vis=False``.
    panels: list[Callable[[Any], None]] = field(default_factory=list)

    # --- Genesis rigid-solver options ---------------------------------------
    # All optional; ``None`` = "use the Genesis default". Mapped to ``gs.options.RigidOptions``
    # by :class:`~genelab.scene.InteractiveScene`, which only passes ``rigid_options`` to
    # ``gs.Scene`` when at least one is set — so an unset config keeps the historic behaviour
    # byte-for-byte. (Genesis exposes no continuous-collision-detection / CCD knob.)
    enable_self_collision: bool | None = None  # RigidOptions.enable_self_collision
    enable_joint_limit: bool | None = None  # RigidOptions.enable_joint_limit
    max_collision_pairs: int | None = None  # RigidOptions.max_collision_pairs
    solver_iterations: int | None = None  # RigidOptions.iterations (constraint solver)
    ls_iterations: int | None = None  # RigidOptions.ls_iterations (line search)
    solver_tolerance: float | None = None  # RigidOptions.tolerance
    constraint_timeconst: float | None = (
        None  # RigidOptions.constraint_timeconst (stiffness/damping)
    )
    integrator: str | None = (
        None  # gs.integrator.<name>: Euler / implicitfast / approximate_implicitfast
    )
    # Store per-DOF / per-link model params (kp/kv/frictionloss/damping/armature, mass/
    # inertia) PER-ENV so they can be domain-randomized per environment. Genesis defaults
    # both to False (params shared across the batch), which silently no-ops per-env dof DR
    # like ``mdp.dr.randomize_joint_stiffness_damping`` on implicit-PD actuators. Costs a
    # little memory (per-env copies of the model arrays); enable for sim2real DR.
    batch_dofs_info: bool | None = None  # RigidOptions.batch_dofs_info
    batch_links_info: bool | None = None  # RigidOptions.batch_links_info

    def rigid_options_kwargs(self) -> dict[str, Any]:
        """Map the *set* rigid-solver fields to ``gs.options.RigidOptions`` kwargs.

        Returns only the fields the user set (skips ``None``), keyed by the Genesis
        ``RigidOptions`` parameter names. ``integrator`` stays a string here — the scene
        resolves it to ``gs.integrator.<name>`` so ``configs`` stays Genesis-free. An empty
        dict means the scene won't pass ``rigid_options`` at all.
        """
        mapping: dict[str, Any] = {
            "enable_self_collision": self.enable_self_collision,
            "enable_joint_limit": self.enable_joint_limit,
            "max_collision_pairs": self.max_collision_pairs,
            "iterations": self.solver_iterations,
            "ls_iterations": self.ls_iterations,
            "tolerance": self.solver_tolerance,
            "constraint_timeconst": self.constraint_timeconst,
            "integrator": self.integrator,
            "batch_dofs_info": self.batch_dofs_info,
            "batch_links_info": self.batch_links_info,
        }
        return {k: v for k, v in mapping.items() if v is not None}

    @staticmethod
    def play_retargeted_keys() -> tuple[str, ...]:
        """Override paths the CLI rewrites ``env.`` → ``play_env.`` in play mode.

        The ``--vis`` / ``--gpu`` / ``--steps`` / ``--dt`` shorthand flags expand
        to ``env.simulation.<field>`` overrides. When a task defines a separate
        ``play_env``, ``genelab play TASK --vis`` should target *that* env, so the
        CLI retargets these keys onto ``play_env.simulation.<field>``. Owning the
        list here (rather than as a private constant in ``cli/__init__.py``) keeps
        the set of play-retargetable simulation overrides next to the fields
        themselves.
        """
        return (
            "env.simulation.vis",
            "env.simulation.gpu",
            "env.simulation.steps",
            "env.simulation.dt",
        )


@dataclass
class InteractiveSceneCfg:
    """Composition of an interactive scene: entities + terrain + sensors + viewer plugins."""

    env_spacing: tuple[float, float] = (2.0, 2.0)
    sensors: "tuple[SensorCfg, ...]" = field(default_factory=tuple)
    mouse_interaction: bool = False
    entities: "dict[str, ArticulationCfg | RigidObjectCfg]" = field(default_factory=dict)
    terrain: "TerrainGeneratorCfg | None" = None
    # When False, the default ground plane (spawned when ``terrain is None``) is
    # non-colliding — still rendered, but bodies pass through it. Analytic deformable
    # terrain uses this so feet are governed by the analytic support over a virtual surface,
    # instead of resting on (or floating above) a rigid plane.
    ground_plane_collision: bool = True
    # World z of the default ground plane. Analytic deformable terrain renders it at the
    # foot level (the virtual surface) so the robot visibly stands on the ground rather than
    # appearing to float above a plane at z=0.
    ground_plane_height: float = 0.0
    # When True, ``InteractiveScene._build`` passes
    # ``gs.renderers.BatchRenderer(use_rasterizer=use_rasterizer)`` to ``gs.Scene``.
    # Required for ``CameraSensor`` to produce per-env RGB-D tensors. Linux x86-64 +
    # CUDA only.
    batch_render: bool = False
    # Forwarded to ``gs.renderers.BatchRenderer(use_rasterizer=...)`` when
    # ``batch_render=True``. ``False`` (default) keeps Genesis's raytracer (higher
    # fidelity, slower); ``True`` switches to the rasterizer for substantially faster
    # batched rendering when photorealism isn't required. Ignored when
    # ``batch_render=False`` since no ``BatchRenderer`` is constructed.
    use_rasterizer: bool = False
    # Recordings are registered as Genesis recorders just before ``gs_scene.build``;
    # each entry describes a data source and one or more output sinks (live plots, file
    # writers, video). See :mod:`genelab.recording` for the dataclass surface.
    recordings: "tuple[RecordingCfg, ...]" = field(default_factory=tuple)
    # Per-solver options for non-rigid materials. The scene auto-enables a deformable /
    # fluid solver with Genesis defaults whenever an entity carries a material of that
    # family; set the matching field here (e.g. ``solvers.mpm``) to tune domain bounds or
    # iteration counts instead. An untouched group leaves rigid-only scenes unchanged.
    solvers: SolverOptionsCfg = field(default_factory=SolverOptionsCfg)


@dataclass
class ActionManagerCfg:
    enabled: bool = True


@dataclass
class ObservationManagerCfg:
    enabled: bool = False


@dataclass
class RewardManagerCfg:
    enabled: bool = False


@dataclass
class TerminationManagerCfg:
    enabled: bool = False


@dataclass
class EventManagerCfg:
    enabled: bool = True


@dataclass
class ManagerBasedEnvCfg:
    """Base shape for Isaac Lab-style manager-based environments."""

    device: str = "cuda"
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    scene: InteractiveSceneCfg = field(default_factory=InteractiveSceneCfg)
    actions: ActionManagerCfg = field(default_factory=ActionManagerCfg)
    observations: ObservationManagerCfg = field(default_factory=ObservationManagerCfg)
    rewards: RewardManagerCfg = field(default_factory=RewardManagerCfg)
    terminations: TerminationManagerCfg = field(default_factory=TerminationManagerCfg)
    events: EventManagerCfg = field(default_factory=EventManagerCfg)


@dataclass
class TaskCfg:
    """Registry-facing task configuration.

    ``env`` intentionally accepts an arbitrary dataclass so examples and downstream projects can
    plug in their own environment configs without changing the core registry or CLI. ``play_env``
    optionally provides a play-mode variant (curriculum off, push off). ``agent`` carries an
    RL runner cfg (e.g. ``RslRlOnPolicyRunnerCfg``) when the task is trainable.
    """

    name: str
    env_name: str
    robot_name: str
    env: object
    play_env: object | None = None
    agent: object | None = None
    trainable: bool = False


_PATH_ALIASES = {
    "vis": "env.simulation.vis",
    "gpu": "env.simulation.gpu",
    "steps": "env.simulation.steps",
    "dt": "env.simulation.dt",
}


def resolve_override_alias(raw_path: str) -> str:
    """Return the dotted path that ``raw_path`` resolves to via the alias table."""
    return _PATH_ALIASES.get(raw_path, raw_path)


def apply_overrides(cfg: object, overrides: dict[str, str]) -> None:
    """Apply dotted CLI overrides to a dataclass config object."""

    for raw_path, raw_value in overrides.items():
        _set_dotted_value(cfg, resolve_override_alias(raw_path), raw_value)


def _set_dotted_value(root: object, dotted_path: str, raw_value: str) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override path: {dotted_path!r}")
    target = root
    for part in parts[:-1]:
        target = _descend(target, part, dotted_path)
    field_name = parts[-1]
    if isinstance(target, dict):
        if field_name not in target:
            raise ValueError(f"unknown override path: {dotted_path!r}")
        current = target[field_name]
        annotation = None
        target[field_name] = _coerce_value(raw_value, current, annotation)
        return
    if not hasattr(target, field_name):
        raise ValueError(f"unknown override path: {dotted_path!r}")
    current = getattr(target, field_name)
    annotation = _field_annotation(target, field_name)
    setattr(target, field_name, _coerce_value(raw_value, current, annotation))


def _descend(target: object, key: str, dotted_path: str) -> object:
    if isinstance(target, dict):
        if key not in target:
            raise ValueError(f"unknown override path: {dotted_path!r}")
        return target[key]
    if not hasattr(target, key):
        raise ValueError(f"unknown override path: {dotted_path!r}")
    return getattr(target, key)


def _field_annotation(obj: object, field_name: str) -> _Annotation | None:
    if not is_dataclass(obj):
        return None
    try:
        hints = get_type_hints(type(obj))
    except NameError:
        # Quoted annotations (e.g. ``"tuple[SensorCfg, ...]"``) only resolve once
        # the referenced module is imported. Coercion falls back to ``type(current)``
        # in this case, which is correct for every override path realistically used.
        return None
    return cast(_Annotation | None, hints.get(field_name))


def _coerce_value(raw_value: str, current: object, annotation: _Annotation | None) -> object:
    if raw_value.lower() in {"none", "null"}:
        return None
    target_type = _strip_optional(annotation)
    if target_type in (None, Any):
        target_type = type(current) if current is not None else str
    origin = get_origin(target_type)
    if target_type is bool or isinstance(current, bool):
        return _parse_bool(raw_value)
    if target_type is int or isinstance(current, int) and not isinstance(current, bool):
        return int(raw_value)
    if target_type is float or isinstance(current, float):
        return float(raw_value)
    if target_type is Path or isinstance(current, Path):
        return Path(raw_value)
    if origin in (tuple, list):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        item_types = get_args(target_type)
        if origin is tuple and item_types:
            if len(item_types) == 2 and item_types[1] is Ellipsis:
                coerced = tuple(_coerce_scalar(value, item_types[0]) for value in values)
            else:
                coerced = tuple(
                    _coerce_scalar(value, typ)
                    for value, typ in zip(values, item_types, strict=True)
                )
            return coerced
        item_type = item_types[0] if item_types else str
        return [_coerce_scalar(value, item_type) for value in values]
    return _coerce_scalar(raw_value, target_type)


def _strip_optional(annotation: _Annotation | None) -> _Annotation | None:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is not UnionType:
        return annotation
    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if len(args) == 1 and len(args) != len(get_args(annotation)):
        return args[0]
    return annotation


def _coerce_scalar(raw_value: str, target_type: _Annotation | None) -> object:
    if target_type is bool:
        return _parse_bool(raw_value)
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    if target_type is Path:
        return Path(raw_value)
    return raw_value


def _parse_bool(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {raw_value!r}")
