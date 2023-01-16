REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .shared_but_sides_controller import SharedButSidesMAC
from .single_controller import SingleAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY['shared_but_sides_mac'] = SharedButSidesMAC
REGISTRY['single_mac'] = SingleAC