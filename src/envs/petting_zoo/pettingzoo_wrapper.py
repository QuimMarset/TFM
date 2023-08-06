from envs.multiagentenv import MultiAgentEnv
from envs.petting_zoo.pistonball.pistonball_wrapper import PistonballWrapper



class PettingZooWrapper(MultiAgentEnv):

    def __init__(self, env_name, **kwargs):
        self.env = self._create_env(env_name, **kwargs)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env.episode_limit


    def _create_env(self, env_name, **kwargs):
        if env_name == 'pistonball':
            return PistonballWrapper(**kwargs)
        else:
            raise ValueError(f'Unknown PettingZoo environment {env_name}')


    def reset(self):
        return self.env.reset()


    def step(self, actions):
        return self.env.step(actions)

    
    def render(self):
        self.env.render()


    def close(self):
        self.env.close()


    def get_stats(self):
        return self.env.get_stats()


    def get_action_shape(self):
        return self.env.get_action_shape()
    

    def get_action_dtype(self):
        return self.env.get_action_dtype()
    

    def get_number_of_discrete_actions(self):
        return self.env.get_number_of_discrete_actions()
    

    def has_discrete_actions(self):
        return self.env.has_discrete_actions()
    

    def get_action_spaces(self):
        return self.env.get_action_spaces()
    

    def get_obs(self):
        return self.env.get_obs()


    def get_obs_shape(self):
        return self.env.get_obs_shape()
        

    def get_state(self):
        return self.env.get_state()


    def get_state_shape(self):
        return self.env.get_state_shape()