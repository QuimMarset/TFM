from controllers.single_agent.dqn_controller import DQNController
from controllers.single_agent.ddpg_controller import DDPGController
from controllers.single_agent.td3_controller import TD3Controller

from controllers.multi_agent.QMIX.q_controller import QController
from controllers.multi_agent.QMIX.continuous_q_controller import ContinuousQController

from controllers.multi_agent.Others.maddpg_discrete_controller import MADDPGDiscreteController
from controllers.multi_agent.Others.facmac_controller import FACMACAgentController
from controllers.multi_agent.Others.jad3_controller import JAD3Controller

from controllers.multi_agent.TransfQMIX.transformer_controller import TransformerController
from controllers.multi_agent.TransfQMIX.transformer_controller_continuous import TransformerContinuousController



REGISTRY = {}

REGISTRY['dqn_controller'] = DQNController
REGISTRY['ddpg_controller'] = DDPGController
REGISTRY['td3_controller'] = TD3Controller

REGISTRY["basic_controller"] = QController
REGISTRY['comix_controller'] = ContinuousQController

REGISTRY['maddpg_discrete_controller'] = MADDPGDiscreteController

REGISTRY['facmac_controller'] = FACMACAgentController

REGISTRY['jad3_controller'] = JAD3Controller

REGISTRY['transformer_continuous_controller'] = TransformerContinuousController
REGISTRY['transformer_controller'] = TransformerController
