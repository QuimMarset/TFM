from learners.q_learner import QLearner
#from .coma_learner import COMALearner
#from .qtran_learner import QLearner as QTranLearner
#from .actor_critic_learner import ActorCriticLearner
from learners.dqn_learner import DQNLearner
from learners.continuous_q_learner import ContinuousQLearner
from learners.facmac_learner import FACMACLearner
from learners.maddpg_learner import MADDPGLearner
from learners.maddpg_distribution_learner import MADDPGDistributionLearner
from learners.facmac_distribution_learner import FACMACDistributionLearner
from learners.maddpg_discrete_learner import MADDPGDiscreteLearner
from learners.ddpg_learner import DDPGLearner




REGISTRY = {}

REGISTRY["q_learner"] = QLearner
#REGISTRY["coma_learner"] = COMALearner
#REGISTRY["qtran_learner"] = QTranLearner
#REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY['dqn_learner'] = DQNLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["continuous_q_learner"] = ContinuousQLearner
REGISTRY["maddpg_distribution_learner"] = MADDPGDistributionLearner
REGISTRY["facmac_distribution_learner"] = FACMACDistributionLearner
REGISTRY['maddpg_discrete_learner'] = MADDPGDiscreteLearner
REGISTRY['ddpg_learner'] = DDPGLearner