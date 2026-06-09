"""Built-in robot configurations bundled with GeneLab.

Importing the package registers every entry into :data:`genelab.registry.ROBOTS` as an
import side-effect. Factories stay lazy: ``genelab list robots`` enumerates names and
descriptions without downloading anything; the actual MJCF / URDF fetch happens only
when ``ROBOTS.get(name)()`` is called (e.g. ``genelab info <name>`` or downstream task
construction).
"""

from genelab.asset_zoo.allegro_hand import AllegroHandCfg
from genelab.asset_zoo.anymal_c import AnymalCCfg
from genelab.asset_zoo.cartpole import CartpoleCfg
from genelab.asset_zoo.franka import FrankaPandaCfg
from genelab.asset_zoo.sample_arm import SampleArmCfg
from genelab.asset_zoo.unitree_g1 import UnitreeG1Cfg
from genelab.asset_zoo.unitree_g1_motions import g1_lafan1_dance1_subject2
from genelab.asset_zoo.unitree_go1 import UnitreeGo1Cfg
from genelab.asset_zoo.unitree_go2w import UnitreeGo2WCfg
from genelab.asset_zoo.unitree_h1 import UnitreeH1Cfg
from genelab.asset_zoo.ur10e import UR10eCfg
from genelab.asset_zoo.wuji_hand import WujiHandCfg

__all__ = [
    "AllegroHandCfg",
    "AnymalCCfg",
    "CartpoleCfg",
    "FrankaPandaCfg",
    "SampleArmCfg",
    "UR10eCfg",
    "UnitreeG1Cfg",
    "UnitreeGo1Cfg",
    "UnitreeGo2WCfg",
    "UnitreeH1Cfg",
    "WujiHandCfg",
    "g1_lafan1_dance1_subject2",
]
