from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
import numpy as np
import time

from collections import UserDict

import gym
import gym.envs.registration

# Do this before importing pybullet_envs (adds an extra property env_specs as a property to the registry, so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry


from pybullet_envs.gym_locomotion_envs import AntBulletEnv


if __name__ == '__main__':


    env = AntBulletEnv(render=True)
    env.reset()

    for i in range(5000):

        action = env.action_space.sample()
        action[4:6] = 0

        env.render()

        state, reward, done, _ = env.step(action)

        if done:
            print(f'Reset at {i}')
            env.reset()

        time.sleep(0.01)

    print()