import gym
import numpy as np



class FloatScaleWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.transform_obs_spaces()
        self.transform_state_space()


    def observation(self, observations):
        return np.array(observations).astype(np.float32) / 255.0


    def state(self):
        state = self.env.state()
        return np.array(state).astype(np.float32) / 255.0

    
    def __transform_space(self, space):
        return gym.spaces.Box(
            low = 0,
            high = 1.0,
            shape = space.shape,
            dtype = np.float32
        )

    def transform_obs_spaces(self):
        obs_spaces = tuple()
        for obs_space in self.observation_space:
            new_obs_space = self.__transform_space(obs_space)
            obs_spaces += (new_obs_space, )
        self.observation_space = obs_spaces


    def transform_state_space(self):
        self.state_space = self.__transform_space(self.state_space)