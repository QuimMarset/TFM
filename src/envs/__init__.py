from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.pettingzoo_wrapper import PettingZooWrapper, PettingZooContinuousWrapper
from envs.gym_ma_wrapper import _GymmaWrapper
from envs.particle import Particle
from envs.multiagent_mujoco import MujocoMulti



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooWrapper)
REGISTRY["pettingzoo_continuous"] = partial(env_fn, env=PettingZooContinuousWrapper)
REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
REGISTRY["particle"] = partial(env_fn, env=Particle)
#REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
#REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)