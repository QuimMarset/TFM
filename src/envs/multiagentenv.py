

class MultiAgentEnv:

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_shape(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_shape(self):
        """ Returns the shape of the state"""
        raise NotImplementedError
    
    def get_number_of_discrete_actions(self):
        """ 
            Ignore it when working with continuous actions
            Discrete environments need to override it
        """
        return -1
    
    def get_action_shape(self):
        """ Returns the shape of a single action """
        raise NotImplementedError
    
    def get_action_dtype(self):
        raise NotImplementedError
    
    def has_discrete_actions(self):
        raise NotImplementedError
    
    def get_action_spaces(self):
        raise NotImplementedError

    def reset(self, test_mode=False):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_entity_attributes(self):
        return {
            'n_entities' : self.n_entities,
            'n_entities_obs' : self.n_entities_obs,
            'n_entities_state' : self.n_entities_state,
            'obs_entity_feats' : self.obs_entity_feats,
            'state_entity_feats' : self.state_entity_feats
        }

    def get_env_info(self, args):
        env_info = {
            "state_shape": self.get_state_shape(),
            "obs_shape": self.get_obs_shape(),
            'action_shape': self.get_action_shape(),
            'n_discrete_actions': self.get_number_of_discrete_actions(),
            'action_dtype': self.get_action_dtype(),
            'has_discrete_actions': self.has_discrete_actions(),
            'action_spaces': self.get_action_spaces(),
            'n_agents': self.n_agents,
            'episode_limit': self.episode_limit
        }

        if args.env_args['obs_entity_mode'] or args.env_args['state_entity_mode']:
            env_info.update(self.get_entity_attributes())

        return env_info
