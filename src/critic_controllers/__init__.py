from critic_controllers.factorized_controller import FactorizedCriticController
from critic_controllers.maddpg_controller import MADDPGCriticController
from critic_controllers.maddpg_discrete_controller import MADDPGDiscreteCriticController
from critic_controllers.ddpg_controller import DDPGController



REGISTRY = {}

REGISTRY['factorized_controller'] = FactorizedCriticController
REGISTRY['maddpg_controller'] = MADDPGCriticController
REGISTRY['maddpg_discrete_controller'] = MADDPGDiscreteCriticController
REGISTRY['ddpg_controller'] = DDPGController
