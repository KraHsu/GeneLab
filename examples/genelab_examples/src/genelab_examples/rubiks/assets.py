"""Rubik's-cube asset specs and MJCF generation."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import cast

Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]
Coord = tuple[int, int, int]
Axis = str | int
Layer = str | int

_AXIS_NAMES = ("x", "y", "z")
_COORD_TOKEN = {-1: "n", 0: "c", 1: "p"}


def _default_face_colors() -> dict[str, RGBA]:
    # Standard-ish cube convention: R/L, F/B, U/D.
    return {
        "xp": (0.86, 0.04, 0.04, 1.0),  # right: red
        "xn": (1.00, 0.45, 0.02, 1.0),  # left: orange
        "yp": (0.02, 0.62, 0.20, 1.0),  # front: green
        "yn": (0.02, 0.18, 0.82, 1.0),  # back: blue
        "zp": (0.96, 0.96, 0.90, 1.0),  # up: white
        "zn": (1.00, 0.86, 0.03, 1.0),  # down: yellow
    }


@dataclass(frozen=True)
class RubiksCubeSpec:
    """Physical and visual dimensions for a Genesis Rubik's-cube asset."""

    cubie_size: float = 0.05
    gap: float = 0.0015
    sticker_thickness: float = 0.001
    sticker_inset: float = 0.006
    mass_per_cubie: float = 0.02
    friction: Vec3 = (1.2, 0.02, 0.0001)
    body_rgba: RGBA = (0.02, 0.02, 0.018, 1.0)
    include_hidden_core: bool = True
    weld_cubies: bool = False
    weld_solref: tuple[float, float] = (0.005, 1.0)
    weld_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    face_colors: dict[str, RGBA] = field(default_factory=_default_face_colors)

    def __post_init__(self) -> None:
        if self.cubie_size <= 0:
            raise ValueError("cubie_size must be positive")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.sticker_thickness <= 0:
            raise ValueError("sticker_thickness must be positive")
        if self.sticker_inset < 0:
            raise ValueError("sticker_inset must be non-negative")
        if self.sticker_inset * 2 >= self.cubie_size:
            raise ValueError("sticker_inset leaves no visible sticker area")
        if self.mass_per_cubie <= 0:
            raise ValueError("mass_per_cubie must be positive")
        missing = set(_default_face_colors()) - set(self.face_colors)
        if missing:
            raise ValueError(f"face_colors missing keys: {sorted(missing)}")

    @property
    def spacing(self) -> float:
        return self.cubie_size + self.gap

    @property
    def full_extent(self) -> float:
        return 3 * self.cubie_size + 2 * self.gap

    @property
    def cubie_half_extent(self) -> float:
        return 0.5 * self.cubie_size

    @property
    def sticker_half_extent(self) -> float:
        return 0.5 * (self.cubie_size - 2 * self.sticker_inset)

    @property
    def cubie_inertia(self) -> float:
        # Principal inertia for a solid cube about its center.
        return (1.0 / 6.0) * self.mass_per_cubie * self.cubie_size**2

    @property
    def cubie_count(self) -> int:
        return sum(1 for _ in iter_cubie_coords(self.include_hidden_core))

    @property
    def sticker_count(self) -> int:
        return sum(
            1 for coord in iter_cubie_coords(self.include_hidden_core) for _ in exposed_faces(coord)
        )


def iter_cubie_coords(include_hidden_core: bool = True) -> tuple[Coord, ...]:
    """Return coordinates for the 3x3x3 cubie lattice."""

    coords = cast(tuple[Coord, ...], tuple(product((-1, 0, 1), repeat=3)))
    if include_hidden_core:
        return coords
    return tuple(coord for coord in coords if coord != (0, 0, 0))


def cubie_name(coord: Coord) -> str:
    x, y, z = coord
    return f"cubie_x{_COORD_TOKEN[x]}_y{_COORD_TOKEN[y]}_z{_COORD_TOKEN[z]}"


def cubie_center(coord: Coord, spec: RubiksCubeSpec) -> Vec3:
    return cast(Vec3, tuple(v * spec.spacing for v in coord))


def exposed_faces(coord: Coord) -> tuple[tuple[int, int, str], ...]:
    """Return exposed sticker faces as ``(axis_index, sign, key)`` tuples."""

    faces: list[tuple[int, int, str]] = []
    for axis, value in enumerate(coord):
        if value == 1:
            faces.append((axis, 1, f"{_AXIS_NAMES[axis]}p"))
        elif value == -1:
            faces.append((axis, -1, f"{_AXIS_NAMES[axis]}n"))
    return tuple(faces)


def sticker_pose(axis: int, sign: int, spec: RubiksCubeSpec) -> tuple[Vec3, Vec3]:
    """Return local MJCF ``pos`` and half-size for a face sticker geom."""

    offset = sign * (spec.cubie_half_extent + 0.5 * spec.sticker_thickness)
    pos = [0.0, 0.0, 0.0]
    size = [spec.sticker_half_extent, spec.sticker_half_extent, spec.sticker_half_extent]
    pos[axis] = offset
    size[axis] = 0.5 * spec.sticker_thickness
    return cast(Vec3, tuple(pos)), cast(Vec3, tuple(size))


def to_mjcf_xml(spec: RubiksCubeSpec | None = None) -> str:
    """Build an MJCF XML string for the cube."""

    spec = spec or RubiksCubeSpec()
    coords = iter_cubie_coords(spec.include_hidden_core)

    root = ET.Element("mujoco", {"model": "rubiks_cube_3x3x3"})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "coordinate": "local",
            "autolimits": "true",
            "inertiafromgeom": "false",
        },
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {
            "friction": _fmt(spec.friction),
            "solref": "0.005 1",
            "solimp": "0.9 0.95 0.001",
            "contype": "1",
            "conaffinity": "0",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    for coord in coords:
        name = cubie_name(coord)
        body = ET.SubElement(
            worldbody, "body", {"name": name, "pos": _fmt(cubie_center(coord, spec))}
        )
        ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": _fmt_scalar(spec.mass_per_cubie),
                "diaginertia": _fmt((spec.cubie_inertia,) * 3),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_body",
                "type": "box",
                "size": _fmt((spec.cubie_half_extent,) * 3),
                "rgba": _fmt(spec.body_rgba),
            },
        )

        for axis, sign, face_key in exposed_faces(coord):
            pos, size = sticker_pose(axis, sign, spec)
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{name}_sticker_{face_key}",
                    "type": "box",
                    "pos": _fmt(pos),
                    "size": _fmt(size),
                    "rgba": _fmt(spec.face_colors[face_key]),
                    "contype": "0",
                    "conaffinity": "0",
                    "density": "0",
                },
            )

    if spec.weld_cubies and len(coords) > 1:
        equality = ET.SubElement(root, "equality")
        anchor = cubie_name((0, 0, 0)) if spec.include_hidden_core else cubie_name(coords[0])
        for coord in coords:
            name = cubie_name(coord)
            if name == anchor:
                continue
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"weld_{name}",
                    "body1": anchor,
                    "body2": name,
                    "solref": _fmt(spec.weld_solref),
                    "solimp": _fmt(spec.weld_solimp),
                },
            )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def write_mjcf(path: str | Path, spec: RubiksCubeSpec | None = None) -> Path:
    """Write the cube MJCF file and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_mjcf_xml(spec), encoding="utf-8")
    return output


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(_fmt_scalar(value) for value in values)


def _fmt_scalar(value: float) -> str:
    return f"{value:.9g}"
