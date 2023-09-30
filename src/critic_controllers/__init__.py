from critic_controllers.single_agent.ddpg_critic_controller import DDPGCriticController
from critic_controllers.single_agent.td3_critic_controller import TD3CriticController

from critic_controllers.multi_agent.maddpg_critic_controller import MADDPGCriticController
from critic_controllers.multi_agent.maddpg_discrete_critic_controller import MADDPGDiscreteCriticController
from critic_controllers.multi_agent.factorized_critic_controller import FactorizedCriticController
from critic_controllers.multi_agent.jad3_critic_controller import JAD3CriticController



REGISTRY = {}

REGISTRY['maddpg_critic_controller'] = MADDPGCriticController
REGISTRY['maddpg_discrete_critic_controller'] = MADDPGDiscreteCriticController

REGISTRY['ddpg_critic_controller'] = DDPGCriticController
REGISTRY['td3_critic_controller'] = TD3CriticController

REGISTRY['factorized_critic_controller'] = FactorizedCriticController
REGISTRY['jad3_critic_controller'] = JAD3CriticController
