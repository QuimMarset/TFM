import gym
import numpy as np
from collections import deque



class StackingWrapper(gym.Wrapper):

    def __init__(self, env, num_stacked):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.num_stacked = num_stacked
        self.obs_stacks = [deque([], maxlen=num_stacked) for _ in range(self.env.num_agents)]
        self.state_stack = deque([], maxlen=num_stacked)
        self.transform_obs_spaces()
        self.transform_state_space()


    def __transform_space(self, space):
        return gym.spaces.Box(
            low = np.repeat(space.low, self.num_stacked, axis=-1),
            high = np.repeat(space.high, self.num_stacked, axis=-1),
            shape = (*space.shape[:2], self.num_stacked),
            dtype = space.dtype
        )

    def transform_obs_spaces(self):
        obs_spaces = tuple()
        for obs_space in self.observation_space:
            new_obs_space = self.__transform_space(obs_space)
            obs_spaces += (new_obs_space, )
        self.observation_space = obs_spaces


    def transform_state_space(self):
        self.state_space = self.__transform_space(self.state_space)


    def __get_observations_from_stacks(self):
        observations = tuple()
        for stack in self.obs_stacks:
            obs = np.stack(stack, axis=-1)
            observations += (obs, )
        return observations

    
    def __add_observations_to_stacks(self, observations):
        for obs, stack in zip(observations, self.obs_stacks):    
            stack.append(np.squeeze(obs, axis=-1))


    def __add_state_to_stack(self):
        state = self.env.state()
        self.state_stack.append(np.squeeze(state, axis=-1))

    
    def __reset_stacks(self, state, observations):
        for _ in range(self.num_stacked):
            self.state_stack.append(np.squeeze(state, axis=-1))
            self.__add_observations_to_stacks(observations)


    def reset(self):
        observations = self.env.reset()
        state = self.env.state()
        self.__reset_stacks(state, observations)
        return self.__get_observations_from_stacks()


    def state(self):
        return np.stack(self.state_stack, axis=-1)


    def step(self, actions):
        observations, rewards, done, info = self.env.step(actions)
        self.__add_observations_to_stacks(observations)
        self.__add_state_to_stack()
        new_observations = self.__get_observations_from_stacks()
        return new_observations, rewards, done, info
        