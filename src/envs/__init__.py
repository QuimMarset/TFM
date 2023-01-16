from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.pettingzoo_wrapper import PettingZooWrapper
from envs.gym_ma_wrapper import _GymmaWrapper


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooWrapper)