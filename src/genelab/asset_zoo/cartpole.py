"""Cartpole asset zoo entry — classic cart-on-rail with a hinged pole on top.

Mirrors the underactuated topology of ``examples/inverted_pendulum``: the cart slides
along one axis under an implicit-PD actuator while the pole hinge is left passive. Use
this entry as the template when adding a new robot config — every other module under
:mod:`genelab.asset_zoo` follows the same shape (declarative :class:`AssetSpec` plus a
lazy factory that calls :func:`fetch_asset` only when invoked).
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

# MJCF lives in the external ``KraHsu/genelab-assets`` repository (see asset zoo concept
# page). The md5 is the digest of the uploaded blob — refresh after every new commit to
# the assets repo so cached copies invalidate cleanly.
_MJCF: Final = AssetSpec(
    name="cartpole",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/cartpole/cartpole.xml",
    md5="44a6aadec0d60f185c53c5086c37675b",
    filename="cartpole.xml",
)

_CART_JOINT: Final = "cart_slide"
_POLE_JOINT: Final = "pole_hinge"
_INIT_BASE_HEIGHT: Final = 0.12


def CartpoleCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the cartpole robot.

    Triggers an md5-verified download on first call, then resolves to the cached MJCF
    on every subsequent call. The returned dataclass is independent per call so callers
    may freely mutate fields (e.g. ``init_pos``) without leaking state across envs.
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, _INIT_BASE_HEIGHT),
        default_joint_pos={_CART_JOINT: 0.0, _POLE_JOINT: 0.0},
        actuators={
            "cart": ImplicitPDActuatorCfg(
                target_names_expr=(_CART_JOINT,),
                stiffness=80.0,
                damping=8.0,
                action_scale=1.0,
            ),
            "pole": ImplicitPDActuatorCfg(
                target_names_expr=(_POLE_JOINT,),
                stiffness=0.0,
                damping=0.0,
                action_scale=0.0,
            ),
        },
    )


register_robot(
    "cartpole",
    CartpoleCfg,
    description="Classic cart on a rail with a passive hinged pole; single actuated slider.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info cartpole",
        "genelab list robots",
    ],
)
