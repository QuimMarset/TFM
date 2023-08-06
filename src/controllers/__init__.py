from controllers.single_agent.dqn_controller import DQNController
from controllers.single_agent.ddpg_controller import DDPGController
from controllers.single_agent.td3_controller import TD3Controller

from controllers.multi_agent.QMIX.q_controller import QController
from controllers.multi_agent.QMIX.continuous_q_controller import ContinuousQController

from controllers.multi_agent.MADDPG.maddpg_discrete_controller import MADDPGDiscreteController

from controllers.multi_agent.FACMAC.facmac_controller import FACMACAgentController
from controllers.multi_agent.FACMAC.facmac_controller_no_rnn import FACMACAgentControllerNoRNN
from controllers.multi_agent.FACMAC.facmac_td3_controller_no_rnn import FACMACTD3AgentControllerNoRNN

from controllers.multi_agent.TransfQMIX.transformer_controller import TransformerController
from controllers.multi_agent.TransfQMIX.transformer_controller_continuous import TransformerContinuousController



REGISTRY = {}

REGISTRY['dqn_mac'] = DQNController
REGISTRY['ddpg_mac'] = DDPGController
REGISTRY['td3_mac'] = TD3Controller

REGISTRY["basic_mac"] = QController
REGISTRY['comix_mac'] = ContinuousQController

REGISTRY['maddpg_discrete_mac'] = MADDPGDiscreteController

REGISTRY['facmac_mac'] = FACMACAgentController
REGISTRY['facmac_no_rnn_mac'] = FACMACAgentControllerNoRNN
REGISTRY['facmac_td3_no_rnn_mac'] = FACMACTD3AgentControllerNoRNN

REGISTRY['transformer_continuous_mac'] = TransformerContinuousController
REGISTRY['transformer_mac'] = TransformerController
