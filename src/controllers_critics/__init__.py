REGISTRY = {}

from .facmac_critic_non_shared_controller import FACMACCriticNonSharedController
from .facmac_critic_type_sides import FACMACCriticController

REGISTRY["critic_non_shared"] = FACMACCriticNonSharedController
REGISTRY["critic_sides"] = FACMACCriticController