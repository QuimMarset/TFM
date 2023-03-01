REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .shared_but_sides_controller import SharedButSidesMAC
from .single_controller import SingleAC
from .cqmix_controller import CQMixMAC
from .cqmix_controller_sides import CQMixMACSides
from .cqmix_controller_non_shared import CQMixNonSharedMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY['shared_but_sides_mac'] = SharedButSidesMAC
REGISTRY['single_mac'] = SingleAC
REGISTRY['cqmix_mac'] = CQMixMAC
REGISTRY['cqmix_sides_mac'] = CQMixMACSides
REGISTRY['cqmix_non_shared_mac'] = CQMixNonSharedMAC
