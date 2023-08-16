import sys
import os
sys.path.insert(0, f'{os.path.abspath("./src")}/envs/base_libraries')


from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pettingzoo_wrapper import PettingZooWrapper
from envs.particle import Particle
from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
#from envs.multiagent_mujoco_farama.mujoco_multi_wrapper import MujocoMultiFaramaWrapper
#from envs.adaptive_optics.ao_wrapper import AdaptiveOpticsWrapper



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["pettingzoo_continuous"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
#REGISTRY["mujoco_multi_farama"] = partial(env_fn, env=MujocoMultiFaramaWrapper)
REGISTRY["particle"] = partial(env_fn, env=Particle)
#REGISTRY["adaptive_optics"] = partial(env_fn, env=AdaptiveOpticsWrapper)
