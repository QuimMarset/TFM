from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pettingzoo_wrapper import PettingZooWrapper
from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.multiagent_mujoco.ant_multi_direction import AntMultiDirectionMultiAgentEnv
#from envs.multiagent_mujoco_farama.mujoco_multi_wrapper import MujocoMultiFaramaWrapper



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["pettingzoo_continuous"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
REGISTRY["ant_multi_direction"] = partial(env_fn, env=AntMultiDirectionMultiAgentEnv)

# We tried the version the people of Farama maintains, but we did not use it in our experiments
#REGISTRY["mujoco_multi_farama"] = partial(env_fn, env=MujocoMultiFaramaWrapper)

# We tried training a method to solve this environment, but it did not work. Do not use it
#REGISTRY["adaptive_optics"] = partial(env_fn, env=AdaptiveOpticsWrapper)
