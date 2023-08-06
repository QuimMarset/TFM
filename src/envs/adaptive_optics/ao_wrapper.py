from gym.spaces import Box
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.adaptive_optics.env import AoEnv



class MaskedBox(Box):

    def __init__(self, mask, low, high, shape=None, dtype=np.float32, seed=None):
        super().__init__(low, high, shape, dtype, seed)
        self.mask = mask

    
    def sample(self, mask=None):
        action = super().sample(mask)
        action = action * self.mask
        return action



class AdaptiveOpticsWrapper(MultiAgentEnv):

    # Agents observe the whole state as they are the actuators of the deformable 
    # mirror and the information is available

    def __init__(self, partition, episode_limit, seed, device_compass=-1, partial_observability=False, **config_params):
        self._check_correct_partition(partition)
                
        parameter_file = f'{config_params["parameter_file"]}.py'
        config_params['parameter_file'] = parameter_file
        config_params.pop('obs_entity_mode')
        config_params.pop('state_entity_mode')
        self.env = AoEnv(config_params, parameter_file, seed, device_compass)

        self._check_num_actuators_are_divisible(partition)

        self.episode_limit = episode_limit
        self.n_agents = partition
        self.seed = seed
        self.num_current_steps = 0
        self.partial_observability = partial_observability

        self._build_agents_masks()
        self._build_action_spaces()
        self.reset(add_one_to_seed=False)


    def _check_correct_partition(self, n_agents):
        if n_agents not in [1, 2, 4]:
            raise ValueError(f'AO environment does not accept {n_agents} agents')
        

    def _check_num_actuators_are_divisible(self, n_agents):
        num_actuators_per_agent = self.env.action_1d_shape[0] / n_agents
        if not num_actuators_per_agent.is_integer():
            raise ValueError(f'Number of actuators is not divisible by the number of agents in AO environment')
        

    def _get_agent_action_space_shape(self):
        side = self.env.action_2d_shape[0] // 2
        if self.n_agents == 2:
            return (self.env.action_2d_shape[0], side + 1)
        elif self.n_agents == 4:
            return (side + 1, side + 1)
        else: # n_agents = 1
            return self.env.action_2d_shape
    

    def _build_action_spaces(self):
        self.action_shape = self._get_agent_action_space_shape()
        self.action_spaces = []

        for i in range(self.n_agents):
            space = MaskedBox(self.agent_masks[i], -1, 1, self.action_shape, 
                              self.env.action_space.dtype)
            self.action_spaces.append(space)


    def reset(self, add_one_to_seed=True):
        self.num_current_steps = 0
        self.state = self.env.reset(add_one_to_seed=add_one_to_seed)
        return self.state
    

    def _build_four_agents_mask(self):
        """
        Assuming the 11x11 actuators matrix
        The first agent's mask would look like this:
            0 0 0 0 1 1
            0 0 1 1 1 1
            0 1 1 1 1 1
            0 1 1 1 1 1
            1 1 1 1 1 1
            0 0 0 0 0 0
        For the other agents, we can simply rotate it clockwise considering:
            A1 | A2
            A4 | A3
        """
        half_side = self.env.action_2d_shape[0] // 2 + 1
        agent_mask = self.env.mask_valid_actuators[:half_side, :half_side].copy()
        agent_mask[-1, :] = 0
        return agent_mask
    

    def _build_two_agents_mask(self):
        """
        Assuming the 11x11 actuators matrix
        The first agent's mask would look like this:
            0 0 0 0 1 1
            0 0 1 1 1 1
            0 1 1 1 1 1
            0 1 1 1 1 1
            1 1 1 1 1 1
            1 1 1 1 1 0
            1 1 1 1 1 0
            0 1 1 1 1 0
            0 1 1 1 1 0
            0 0 1 1 1 0
            0 0 0 0 1 0
        For the other agent, we can simply rotate it clockwise considering:
            A1 | A2
        """
        half_side = self.env.action_2d_shape[0] // 2
        agent_mask = self.env.mask_valid_actuators[:, :half_side + 1].copy()
        agent_mask[half_side:, half_side] = 0
        return agent_mask
    

    def _build_agents_masks(self):
        if self.n_agents == 2:
            angle_multiplier = 2
            mask = self._build_two_agents_mask()
        elif self.n_agents == 4:
            angle_multiplier = 1
            mask = self._build_four_agents_mask()
        else:
            angle_multiplier = 1
            mask = self.env.mask_valid_actuators

        self.agent_masks = []
        for i in range(self.n_agents):
            # clockwise rotation
            mask_i = np.rot90(mask, -i*angle_multiplier, axes=(0, 1))
            self.agent_masks.append(mask_i)

        self.agent_masks = np.array(self.agent_masks, dtype=np.float32)
    

    def get_agent_masks(self):
        return self.agent_masks


    def step(self, actions):
        # Apply masks again to ensure we remove wrong actuators
        # It can happen in methods that add noise to the actions
        actions = actions * self.agent_masks       

        if self.n_agents > 1:
            actions = self._build_joint_actions(actions)        
        
        self.state, reward, done, info = self.env.step(actions, controller_type='RL')
        self.num_current_steps += 1
        terminated = self.num_current_steps >= self.episode_limit
        return reward, done, terminated, info
    

    def _get_agent_actuator_indices(self, agent_index):
        half_side = self.env.action_2d_shape[0] // 2

        if self.n_agents == 1:
            start_row = start_col = 0
            end_row = end_col = self.env.action_2d_shape[0]

        elif self.n_agents == 2:
            start_row = 0
            end_row = self.env.action_2d_shape[0]
            start_col = 0 if agent_index == 0 else half_side
            end_col = half_side + 1 if agent_index == 0 else self.env.action_2d_shape[0]

        else: # n_agents == 4
            start_row = 0 if agent_index in [0, 1] else half_side
            end_row = start_row + half_side + 1
            start_col = 0 if agent_index in [0, 3] else half_side
            end_col = start_col + half_side + 1
        
        return start_row, end_row, start_col, end_col
    

    def _build_joint_actions(self, actions):
        joint_actions = np.zeros(self.env.action_2d_shape)

        for i in range(self.n_agents):
            start_row, end_row, start_col, end_col = self._get_agent_actuator_indices(i)
            joint_actions[start_row:end_row, start_col:end_col] += actions[i]

        return joint_actions

    
    def render(self):
        pass


    def close(self):
        pass


    def get_stats(self):
        return {}


    def get_action_shape(self):
        return self.action_spaces[0].shape
    

    def get_action_dtype(self):
        return self.env.action_space.dtype
    

    def has_discrete_actions(self):
        return False
    

    def get_action_spaces(self):
        return self.action_spaces
    

    def _get_partial_observability_observation(self, agent_index):
        state = self.get_state()
        start_row, end_row, start_col, end_col = self._get_agent_actuator_indices(agent_index)
        obs = state[:, start_row:end_row, start_col:end_col]
        return obs * self.agent_masks[agent_index]


    def get_obs(self):
        state = self.get_state()
        obs = []
        for i in range(self.n_agents):
            if not self.partial_observability:
                obs.append(state)
            else:
                obs.append(self._get_partial_observability_observation(i))
        return np.array(obs, dtype=np.float32)


    def get_obs_shape(self):
        return self.get_obs()[0].shape
        

    def get_state(self):
        return self.state


    def get_state_shape(self):
        return self.get_state().shape


    def get_env_info(self, args):
        env_info = super().get_env_info(args)
        env_info['agent_masks'] = self.agent_masks
        return env_info