from envs.petting_zoo.pistonball.pistonball_positions_global_reward_2_actions import parallel_env
from envs.petting_zoo.pistonball.pistonball_test import parallel_env as parallel_env_2
from envs.multiagentenv import MultiAgentEnv



class PistonballWrapper(MultiAgentEnv):


    def __init__(self, num_pistons, continuous, **kwargs):
        time_penalty = kwargs.get('time_penalty', -0.1) 
        episode_limit = kwargs.get('episode_limit', 125) 
        render_mode = kwargs.get('render_mode', None) 
        state_entity_mode = kwargs.get('state_entity_mode', False)

        if kwargs.get('use_flag_version', False):
            self.env = parallel_env_2(n_pistons=num_pistons, continuous=continuous, 
                    time_penalty=time_penalty, max_cycles=episode_limit,
                    render_mode=render_mode, state_entity_mode=state_entity_mode)
        
        else:
            self.env = parallel_env(n_pistons=num_pistons, continuous=continuous, 
                    time_penalty=time_penalty, max_cycles=episode_limit,
                    render_mode=render_mode, state_entity_mode=state_entity_mode)
        
        self.seed = kwargs.get('seed', None)
        self.n_agents = num_pistons
        self.continuous = continuous
        self.episode_limit = episode_limit
        
        if state_entity_mode:
            self._set_entity_attributes()
        
        self.reset()


    def _set_entity_attributes(self):
        self.n_entities = self.env.unwrapped.n_entities
        self.n_entities_obs = self.env.unwrapped.n_entities_obs
        self.obs_entity_feats = self.env.unwrapped.obs_entity_feats
        self.n_entities_state = self.env.unwrapped.n_entities_state
        self.state_entity_feats = self.env.unwrapped.state_entity_feats


    def _transform_dict_to_list(self, value_dict):
        value_list = []
        for agent_id in value_dict:
            value_list.append(value_dict[agent_id])
        return value_list
    

    def _transform_list_to_dict(self, value_list):
        value_dict = {}
        for i, agent_id in enumerate(self.env.agents):
            value_dict[agent_id] = value_list[i]
        return value_dict


    def reset(self, seed=None):
        obs_dict = self.env.reset(seed)
        self.observations = self._transform_dict_to_list(obs_dict)
        return self.observations


    def _preprocess_actions_if_discrete(self, actions):
        if not self.continuous:
            # actions come in a np array with dims (n_agents, 1)
            actions_list = []
            for action in actions:
                actions_list.append(action[0])
            return actions_list
        return actions


    def step(self, actions):
        actions = self._preprocess_actions_if_discrete(actions)
        actions_dict = self._transform_list_to_dict(actions)
        
        observations_dict, rewards_dict, dones_dict, truncations_dict, _ = self.env.step(actions_dict)
        self.observations = self._transform_dict_to_list(observations_dict)
        rewards = self._transform_dict_to_list(rewards_dict)
        dones = self._transform_dict_to_list(dones_dict)
        truncations = self._transform_dict_to_list(truncations_dict)

        return sum(rewards), all(dones), all(truncations), {}

    
    def render(self):
        self.env.render()


    def close(self):
        self.env.close()


    def get_stats(self):
        return {}


    def get_action_shape(self):
        if self.continuous:
            return self.env.action_space(self.env.agents[0]).shape[0]
        return 1
    

    def get_action_dtype(self):
        return self.env.action_space(self.env.agents[0]).dtype
    

    def get_number_of_discrete_actions(self):
        if self.continuous:
            return -1
        return self.env.action_space(self.env.agents[0]).n
    

    def has_discrete_actions(self):
        return not self.continuous
    

    def get_action_spaces(self):
        action_spaces = []
        for agent in self.env.agents:
            action_spaces.append(self.env.action_space(agent))
        return action_spaces
    

    def get_obs(self):
        return self.observations


    def get_obs_shape(self):
        return len(self.get_obs()[0])
        

    def get_state(self):
        return self.env.state()


    def get_state_shape(self):
        return len(self.get_state())
    