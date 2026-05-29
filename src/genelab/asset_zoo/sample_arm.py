"""Bundled xacro showcase — minimal 2-DoF arm demonstrating ``urdf_path`` + ``xacro_args``.

Unlike the other asset zoo entries this robot ships embedded in the GeneLab wheel
under :mod:`genelab.asset_zoo.data` rather than living in the
``KraHsu/genelab-assets`` mirror. The xacro file is tiny (one parameterised arm
length, two revolute joints) and its purpose is purely to demonstrate that
:class:`~genelab.entity.ArticulationCfg`'s ``urdf_path`` + ``xacro_args`` machinery
genuinely round-trips through Genesis 1.0's ``gs.morphs.URDF`` preprocessor.

Larger robots (Go1, G1, Franka, …) stay in the external mirror and arrive via
:func:`~genelab.utils.download.fetch_asset` — the embedded path is the deliberate
shortcut for the sample, not the rule.
"""

from importlib.resources import as_file, files
from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot


_DEFAULT_LINK_LENGTH: Final[float] = 0.3
_XACRO_RESOURCE: Final = files("genelab.asset_zoo.data").joinpath("sample_arm.urdf.xacro")


def _xacro_path() -> str:
    """Materialise the packaged xacro as an on-disk path Genesis can open.

    ``importlib.resources.files`` returns an ``importlib.abc.Traversable`` that may
    point inside a zip / wheel; Genesis's URDF morph needs a real filesystem path so we
    bounce through :func:`importlib.resources.as_file`. The context exits as soon as
    we capture the path, but because the wheel always extracts the data file on first
    import the path remains valid for the lifetime of the process. Tests that run from
    a source checkout (the common case) see the file in place and never enter the
    temp-extract branch.
    """
    with as_file(_XACRO_RESOURCE) as p:
        return str(p)


def SampleArmCfg(link_length: float = _DEFAULT_LINK_LENGTH) -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the bundled sample arm.

    ``link_length`` propagates through ``xacro:arg link_length``: Genesis runs the
    xacro preprocessor on the file, substituting ``${L}`` and the derived
    ``${L * 0.6}`` so the resulting URDF carries link sizes that match the request.
    """
    return ArticulationCfg(
        urdf_path=_xacro_path(),
        xacro_args={"link_length": f"{float(link_length)}"},
        init_pos=(0.0, 0.0, 0.1),
        default_joint_pos={"shoulder_yaw": 0.0, "elbow_pitch": 0.0},
        actuators={
            # Match the two named revolute joints explicitly; Genesis's URDF morph adds
            # a ``root_joint`` for the floating base which must stay outside the actuator
            # group (actuated only by gravity in the default cfg).
            "arm": ImplicitPDActuatorCfg(
                target_names_expr=("shoulder_yaw", "elbow_pitch"),
                stiffness=20.0,
                damping=1.0,
                effort_limit=10.0,
                action_scale=0.5,
            ),
        },
    )


register_robot(
    "sample-arm",
    SampleArmCfg,
    description=(
        "Bundled 2-DoF xacro showcase — demonstrates ArticulationCfg.urdf_path + "
        "xacro_args round-tripping through Genesis 1.0's URDF morph."
    ),
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info sample-arm",
        "python -c 'from genelab.asset_zoo import SampleArmCfg; "
        "print(SampleArmCfg(link_length=0.5))'",
    ],
)
