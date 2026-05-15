"""Built-in robot configurations bundled with GeneLab.

Importing the package registers every entry into :data:`genelab.registry.ROBOTS` as an
import side-effect. Factories stay lazy: ``genelab list robots`` enumerates names and
descriptions without downloading anything; the actual MJCF / URDF fetch happens only
when ``ROBOTS.get(name)()`` is called (e.g. ``genelab info <name>`` or downstream task
construction).
"""

from genelab.asset_zoo.anymal_c import AnymalCCfg
from genelab.asset_zoo.cartpole import CartpoleCfg
from genelab.asset_zoo.franka import FrankaPandaCfg
from genelab.asset_zoo.unitree_g1 import UnitreeG1Cfg
from genelab.asset_zoo.unitree_go1 import UnitreeGo1Cfg

__all__ = ["AnymalCCfg", "CartpoleCfg", "FrankaPandaCfg", "UnitreeG1Cfg", "UnitreeGo1Cfg"]
