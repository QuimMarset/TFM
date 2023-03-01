import gym
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch



def convert_box_space_to_gym(gymnasium_box):
    return Box(gymnasium_box.low, gymnasium_box.high, shape=gymnasium_box.shape, dtype=gymnasium_box.dtype)


def convert_discrete_space_to_gym(gymnasium_discrete):
    return Discrete(gymnasium_discrete.n)



class PettingZooToGymWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = self.env.num_agents
        self.continuous = self.env.continuous
        self.__adapt_observation_space()
        self.__adapt_action_space()
        self.__adapt_state_space()

    
    def __adapt_space(self, space_function, space_converter_function):
        new_spaces = tuple()
        for agent_id in self.env.agent_names:
            agent_space = space_function(agent_id)
            agent_space = space_converter_function(agent_space)
            new_spaces += (agent_space, )
        return new_spaces

    
    def __adapt_observation_space(self):
        self.observation_space = self.__adapt_space(self.env.observation_space, convert_box_space_to_gym)


    def __adapt_action_space(self):
        convert_function = (convert_box_space_to_gym if self.continuous 
            else convert_discrete_space_to_gym)
        self.action_space = self.__adapt_space(self.env.action_space, convert_function)


    def __adapt_state_space(self):
        self.state_space = convert_box_space_to_gym(self.env.state_space)


    def reset(self):
        observations = self.env.reset()
        new_observations = tuple()
        for agent_id in observations:
            new_observations += (observations[agent_id], )
        return new_observations


    def __preprocess_step_actions(self, actions):
        new_actions = {}
        for action, agent_name in zip(actions, self.env.agent_names):
            if isinstance(action, torch.Tensor):
                action = action.item()
            elif isinstance(action, np.ndarray):
                action = action[0]
            new_actions[agent_name] = action
        return new_actions


    def step(self, actions):
        new_observations = tuple()
        new_rewards = []
        new_done = []
        new_info = {}
        actions_dict = self.__preprocess_step_actions(actions)

        observations, rewards, terminations, truncations, info = self.env.step(actions_dict)

        for agent_id in actions_dict:
            new_observations += (observations[agent_id],)
            new_rewards.append(rewards[agent_id])
            new_done.append(terminations[agent_id] or truncations[agent_id])
            new_info[agent_id] = info[agent_id]

        return new_observations, new_rewards, new_done, new_info


    def render(self, mode, **kwargs):
        self.env.render()


    def state(self):
        return self.env.state()


    def get_current_frame(self):
        return self.env.get_current_frame()