import numpy as np
from gym.spaces import flatdim
import pretrained as pretrained
from envs.common_wrappers import TimeLimit, FlattenObservation
from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pistonball import PistonBall
from envs.petting_zoo.pistonball_reward import PistonBallReward
from envs.petting_zoo.pistonball_reward_2_actions import PistonBallReward2Actions
from envs.petting_zoo.wrappers.pettingzoo_to_gym_wrapper import PettingZooToGymWrapper



def piston_ball_creation(**kwargs):
    env = PistonBall(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env


def piston_ball_custom_reward_creation(**kwargs):
    env = PistonBallReward(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env


def piston_ball_custom_reward_2_actions_creation(**kwargs):
    env = PistonBallReward2Actions(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env


def env_creator(key, **kwargs):
    if key == 'pistonball':
        return piston_ball_creation(**kwargs)
    elif key == 'pistonball_reward':
        return piston_ball_custom_reward_creation(**kwargs)
    elif key == 'pistonball_reward_2_actions':
        return piston_ball_custom_reward_2_actions_creation(**kwargs)
    else:
        raise NotImplementedError(f'{key} not yet implemented in EPyMARL')



class PettingZooWrapper(MultiAgentEnv):

    def __init__(self, key, pretrained_wrapper, **kwargs):
        self.initial_env = env_creator(key, **kwargs)
        self.env = TimeLimit(self.initial_env, max_episode_steps=125)
        self.env = FlattenObservation(self.env)
        self.episode_limit = 125

        if pretrained_wrapper:
            self.env = getattr(pretrained, pretrained_wrapper)(self.env)

        self.n_agents = self.initial_env.num_agents
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self.env.observation_space, key=lambda x: x.shape)


    def step(self, actions):
        self.obs, reward, done, _ = self.env.step(actions)
        self.obs = [np.pad(o, (0, self.longest_observation_space.shape[0] - len(o)),
                "constant", constant_values=0) for o in self.obs]
        return float(sum(reward)), all(done), {}


    def get_obs(self):
        return self.obs


    def get_obs_agent(self, agent_id):
        raise self.obs[agent_id]


    def get_obs_size(self):
        return flatdim(self.longest_observation_space)


    def get_state(self):
        return np.reshape(self.initial_env.state(), -1)


    def get_state_size(self):
        return flatdim(self.initial_env.state_space)


    def get_obs_agent(self, agent_id):
        raise self.obs[agent_id]


    def get_obs_size(self):
        return flatdim(self.longest_observation_space)


    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions


    def get_avail_agent_actions(self, agent_id):
        valid = flatdim(self.env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)


    def reset(self):
        self.obs = self.env.reset()
        self.obs = [np.pad(o, (0, self.longest_observation_space.shape[0] - len(o)),
                "constant", constant_values=0) for o in self.obs]
        return self.get_obs(), self.get_state()


    def render(self):
        self.env.render()


    def close(self):
        self.env.close()


    def seed(self):
        return self.env.seed


    def save_replay(self):
        pass


    def get_stats(self):
        return {}

    
    def get_current_frame(self):
        return self.env.get_current_frame()
