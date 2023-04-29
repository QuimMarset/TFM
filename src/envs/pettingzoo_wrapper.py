import numpy as np
from gym.spaces import flatdim
import pretrained as pretrained
from envs.common_wrappers import TimeLimit, FlattenObservation
from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pistonball import PistonBall
from envs.petting_zoo.pistonball_reward import PistonBallReward
from envs.petting_zoo.pistonball_reward_2_actions import PistonBallReward2Actions
from envs.petting_zoo.pistonball_entities import PistonBallEntities
from envs.petting_zoo.pistonball_entities_custom_reward import PistonBallEntitiesCustomReward
from envs.petting_zoo.wrappers.pettingzoo_to_gym_wrapper import PettingZooToGymWrapper
from envs.petting_zoo.pistonball_positions_global_reward_2_actions import PistonBallPositionsGlobalReward2Actions



def piston_ball_creation(**kwargs):
    # Uses positions (from both the pistons and the ball) instead of image frames
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


def pistonball_entities(**kwargs):
    env = PistonBallEntities(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env


def pistonball_entities_custom_reward(**kwargs):
    env = PistonBallEntitiesCustomReward(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env

def pistonball_positions_global_reward_2_actions(**kwargs):
    env = PistonBallPositionsGlobalReward2Actions(**kwargs)
    env = PettingZooToGymWrapper(env)
    return env


def env_creator(key, **kwargs):
    if key == 'pistonball':
        return piston_ball_creation(**kwargs)
    elif key == 'pistonball_reward':
        return piston_ball_custom_reward_creation(**kwargs)
    elif key == 'pistonball_reward_2_actions':
        return piston_ball_custom_reward_2_actions_creation(**kwargs)
    elif key == 'pistonball_entities':
        return pistonball_entities(**kwargs)
    elif key == 'pistonball_entities_custom_reward':
        return pistonball_entities_custom_reward(**kwargs)
    elif key == 'pistonball_positions_global_2_actions':
        return pistonball_positions_global_reward_2_actions(**kwargs)
    else:
        raise NotImplementedError(f'{key} not yet implemented')



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


    def get_obs_shape(self):
        return flatdim(self.longest_observation_space)


    def get_state(self):
        return np.reshape(self.initial_env.state(), -1)


    def get_state_shape(self):
        return flatdim(self.initial_env.state_space)


    def get_obs_agent(self, agent_id):
        raise self.obs[agent_id]


    def get_obs_shape(self):
        return flatdim(self.longest_observation_space)
    

    def get_action_shape(self):
        return 1


    def reset(self):
        self.obs = self.env.reset()
        self.obs = [np.pad(o, (0, self.longest_observation_space.shape[0] - len(o)),
                "constant", constant_values=0) for o in self.obs]
        return self.get_obs(), self.get_state()


    def render(self):
        self.env.render(mode="human")


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
    

    def get_number_of_discrete_actions(self):
        return self.longest_action_space.n

    
    def get_action_dtype(self):
        return self.longest_action_space.dtype
    

    def has_discrete_actions(self):
        return True
    

    def get_action_spaces(self):
        return self.env.action_space



class PettingZooContinuousWrapper(PettingZooWrapper):
    
    def __init__(self, key, pretrained_wrapper=None, **kwargs):
        self.initial_env = env_creator(key, **kwargs)
        self.env = TimeLimit(self.initial_env, max_episode_steps=125)
        self.env = FlattenObservation(self.env)
        self.episode_limit = 125

        if pretrained_wrapper:
            self.env = getattr(pretrained, pretrained_wrapper)(self.env)

        self.n_agents = self.initial_env.num_agents
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.shape)
        self.longest_observation_space = max(self.env.observation_space, key=lambda x: x.shape)


    def step(self, actions):
        self.obs, reward, done, _ = self.env.step(actions)
        return float(sum(reward)), all(done), {}
    

    def get_action_shape(self):
        return self.longest_action_space.shape[0]


    def reset(self):
        self.obs = self.env.reset()
        return self.get_obs(), self.get_state()
    

    def get_number_of_discrete_actions(self):
        return -1
    

    def has_discrete_actions(self):
        return False