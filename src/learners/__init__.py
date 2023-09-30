from learners.single_agent.dqn_learner import DQNLearner
from learners.single_agent.td3_learner import TD3Learner
from learners.single_agent.ddpg_learner import DDPGLearner

from learners.multi_agent.QMIX.q_learner import QLearner
from learners.multi_agent.QMIX.continuous_q_learner import ContinuousQLearner

from learners.multi_agent.Others.iql_continuous_learner import IQLContinuousLearner
from learners.multi_agent.Others.facmac_learner import FACMACLearner
from learners.multi_agent.Others.jad3_learner import JAD3Learner

from learners.multi_agent.MADDPG.maddpg_learner import MADDPGLearner
from learners.multi_agent.MADDPG.maddpg_discrete_learner import MADDPGDiscreteLearner

from learners.multi_agent.TransfQMIX.transformer_learner_discrete import DiscreteTransformerLearner
from learners.multi_agent.TransfQMIX.transformer_learner_continuous import ContinuousTransformerLearner



REGISTRY = {}

REGISTRY['dqn_learner'] = DQNLearner
REGISTRY['ddpg_learner'] = DDPGLearner
REGISTRY['td3_learner'] = TD3Learner

REGISTRY["q_learner"] = QLearner
REGISTRY["continuous_q_learner"] = ContinuousQLearner

REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY['jad3_learner'] = JAD3Learner

REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY['maddpg_discrete_learner'] = MADDPGDiscreteLearner

REGISTRY['iql_continuous_learner'] = IQLContinuousLearner

REGISTRY['discrete_transformer_learner'] = DiscreteTransformerLearner
REGISTRY['continuous_transformer_learner'] = ContinuousTransformerLearner
