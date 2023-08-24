import numpy as np
# Do this before importing pybullet_envs 
# (adds an extra property env_specs as a property to the registry, 
# so it looks like the <0.26 envspec version)
from collections import UserDict
import gym
import gym.envs.registration
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry
from pybullet_envs.gym_locomotion_envs import AntBulletEnv



class PyBulletAntWrapper:


    def __init__(self, render_mode, episode_limit):
        self.episode_limit = episode_limit
        self.render_mode = render_mode
        render = render_mode == 'human'
        self.env = AntBulletEnv(render=render)


    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        self.state = self.env.reset()
        self.step_counter = 0
    

    @property
    def obs(self):
        return self.state
    

    @property
    def observation_space(self):
        return self.env.observation_space
    

    @property
    def action_space(self):
        return self.env.action_space


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_counter += 1
        truncated = self.step_counter >= self.episode_limit

        if self.render_mode == 'human':
            self.render()

        return obs, reward, done, truncated, info


    def render(self):
        self.env.render()


    def close(self):
        self.env.close()
