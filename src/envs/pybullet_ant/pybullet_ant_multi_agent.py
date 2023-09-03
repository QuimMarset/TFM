import numpy as np
from gym.spaces import Box
from envs.multiagentenv import MultiAgentEnv
from envs.pybullet_ant.pybullet_ant_wrapper import PyBulletAntWrapper
from envs.pybullet_ant.pybullet_ant_partitions import create_ant_partition



class PyBulletAntMultiAgent(MultiAgentEnv):


    def __init__(self, num_agents, episode_limit, render_mode, seed, **kwargs):
        self._check_number_of_agents(num_agents)
        self.num_agents = num_agents
        self.episode_limit = episode_limit
        self.seed = seed
        self.render_mode = render_mode
        self.env = PyBulletAntWrapper(render_mode, episode_limit)

        self._create_agents_action_spaces()
        self._create_agents_observation_spaces()
        self.agent_partition = create_ant_partition(num_agents)

        self.reset()


    def _check_number_of_agents(self, num_agents):
        if num_agents not in [1, 2, 4]:
            raise ValueError(f'Can only divide PyBullet Ant into 1, 2, or 4 agents')
        

    def _create_agents_action_spaces(self):
        num_actions = self.env.action_space.shape[0]
        actions_per_agent = num_actions // self.num_agents

        low = self.env.action_space.low[0]
        high = self.env.action_space.high[0]
        self.action_spaces = [
            Box(low, high, shape=(actions_per_agent,), seed=self.seed)
            for _ in range(self.num_agents)
        ]


    def _create_agents_observation_spaces(self):
        num_legs = self.env.action_space.shape[0]
        legs_per_agent = num_legs // self.num_agents
        features_per_agent = 4 * legs_per_agent

        low = self.env.observation_space.low[0]
        high = self.env.observation_space.high[0]
        self.observation_spaces = [
            Box(low, high, shape=(features_per_agent,))
            for _ in range(self.num_agents)
        ]

    
    def get_obs_agent(self, agent_index):
        state = self.env.state
        agent_hinges = self.agent_partition[agent_index]
        obs = []
        for hinge in agent_hinges:
            obs.append(state[hinge.angle_index])
            obs.append(state[hinge.angular_vel_index])
        return obs


    def get_obs(self):
        obs = []
        for i in range(self.num_agents):
            obs_i = self.get_obs_agent(i)
            obs.append(obs_i)
        return np.array(obs)
    

    def step(self, actions):
        num_actions = self.env.action_space.shape[0]
        joint_action = np.zeros(num_actions)

        for i, agent_hinges in enumerate(self.agent_partition):
            for j, hinge in enumerate(agent_hinges):
                joint_action[hinge.hinge_index] = actions[i][j]

        _, reward, done, truncated, info = self.env.step(joint_action)
        return reward, done, truncated, info


    def get_state(self):
        return self.env.state

    
    def get_obs_shape(self):
        return len(self.get_obs_agent(0))


    def get_state_shape(self):
        return len(self.env.state)
    
    
    def get_action_shape(self):
        return self.action_spaces[0].shape[0]
    

    def get_action_dtype(self):
        return np.float32
    

    def has_discrete_actions(self):
        return False
    

    def get_action_spaces(self):
        return self.action_spaces


    def reset(self, test_mode=False):
        self.env.reset(self.seed)
        return self.get_obs()


    def render(self):
        self.env.render()


    def close(self):
        self.env.close()


    def save_replay(self):
        pass
