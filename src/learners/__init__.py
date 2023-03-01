from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner
from .ppo_learner import PPOLearner
from .q_learner_single import QLearnerSingle
from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .facmac_learner_2 import FACMACLearnerRE
from .maddpg_learner import MADDPGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY['q_learner_single'] = QLearnerSingle
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_RE"] = FACMACLearnerRE
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["cq_learner"] = CQLearner