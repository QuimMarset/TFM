import gym
import cv2
import numpy as np



class TransformationWrapper(gym.ObservationWrapper):

    def __init__(self, env, obs_size, state_size, grayscale, **kwargs):
        super().__init__(env)
        self.obs_size = obs_size
        self.state_size = state_size
        self.grayscale = grayscale
        self.num_agents = env.num_agents
        self.__transform_obs_spaces()
        self.__transform_state_space()


    def __transform_image(self, image, new_size):
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, new_size)
        image = np.expand_dims(image, axis=-1)
        return image

    
    def __transform_observations(self, observations):
        new_observations = tuple()
        for obs in observations:
            transformed_obs = self.__transform_image(obs, self.obs_size)
            new_observations += (transformed_obs, )
        return new_observations


    def __transform_space(self, space, new_size):
        num_channels = 1 if self.grayscale else 3
        return gym.spaces.Box(
            low = 0,
            high = 255 if space.dtype == np.uint8 else 1.0,
            shape = (*new_size, num_channels),
            dtype = space.dtype
        )

    def __transform_obs_spaces(self):
        obs_spaces = tuple()
        for obs_space in self.observation_space:
            new_obs_space = self.__transform_space(obs_space, self.obs_size)
            obs_spaces += (new_obs_space, )
        self.observation_space = obs_spaces


    def __transform_state_space(self):
        self.state_space = self.__transform_space(self.state_space, self.state_size)


    def observation(self, observations):
        return self.__transform_observations(observations)


    def state(self):
        state = self.env.state()
        return self.__transform_image(state, self.state_size)