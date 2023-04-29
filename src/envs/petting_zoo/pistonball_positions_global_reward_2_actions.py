from envs.petting_zoo.pistonball_mods.pistonball_positions_global_reward_2_actions import parallel_env
from gym.envs.registration import EnvSpec
import numpy as np



class PistonBallPositionsGlobalReward2Actions:

    def __init__(self, num_pistons, continuous, time_penalty, alpha, render, seed=None, **kwargs):
        self.env = parallel_env(n_pistons=num_pistons, continuous=continuous, time_penalty=time_penalty)
        if render:
            self.env.unwrapped.render_mode = 'human'
        self.continuous = continuous
        self.seed = seed
        self.num_agents = len(self.env.unwrapped.agents)
        self.spec = EnvSpec('Pistonball-v0', 'pettingzoo.butterfly.pistonball:parallel_env')
        self.agent_names = self.env.unwrapped.agents
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space


    def reset(self):
        return self.env.reset(seed=self.seed)

    
    def step(self, actions):
        actions_dict = {}

        if self.continuous:
            for agent_id in actions:
                actions_dict[agent_id] = np.array([actions[agent_id]], dtype=np.float32)
        else:
            actions_dict = actions

        return self.env.step(actions_dict)


    def state(self):
        return self.env.state()


    def close(self):
        self.env.close()


    def render(self):
        self.env.render()

    
    def get_current_frame(self):
        return self.env.aec_env.env.env.get_current_frame()