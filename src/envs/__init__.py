from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pettingzoo_wrapper import PettingZooWrapper
#from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
#from envs.multiagent_mujoco.ant_multi_direction import AntMultiDirectionMultiAgentEnv
#from envs.multiagent_mujoco_farama.mujoco_multi_wrapper import MujocoMultiFaramaWrapper
#from envs.adaptive_optics.ao_wrapper import AdaptiveOpticsWrapper



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["pettingzoo_continuous"] = partial(env_fn, env=PettingZooWrapper)
#REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
#REGISTRY["ant_multi_direction"] = partial(env_fn, env=AntMultiDirectionMultiAgentEnv)
#REGISTRY["mujoco_multi_farama"] = partial(env_fn, env=MujocoMultiFaramaWrapper)
#REGISTRY["adaptive_optics"] = partial(env_fn, env=AdaptiveOpticsWrapper)
