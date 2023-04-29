REGISTRY = {}

from controllers.basic_controller import BasicMAC
from controllers.maddpg_controller import MADDPGMAC
from controllers.dqn_controller import DQNController
from controllers.facmac_controller import FACMACAgentController
from controllers.maddpg_distribution_controller import MADDPGDistributionMAC
from controllers.continuous_q_controller import ContinuousQController
from controllers.facmac_distribution_controller import FACMACDistributionAgentController
from controllers.maddpg_discrete_controller import MADDPGDiscreteController
from controllers.ddpg_controller import DDPGController


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY['dqn_mac'] = DQNController
REGISTRY['facmac_mac'] = FACMACAgentController
REGISTRY['maddpg_distribution_mac'] = MADDPGDistributionMAC
REGISTRY['comix_mac'] = ContinuousQController
REGISTRY['facmac_distribution_mac'] = FACMACDistributionAgentController
REGISTRY['maddpg_discrete_mac'] = MADDPGDiscreteController
REGISTRY['ddpg_mac'] = DDPGController