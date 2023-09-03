import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.multiagent_mujoco.manyagent_ant import ManyAgentAntEnv
from envs.multiagent_mujoco.manyagent_swimmer import ManyAgentSwimmerEnv
from envs.multiagent_mujoco.obsk import get_joints_at_kdist, get_parts_and_edges, build_obs



class MujocoMulti(MultiAgentEnv):

    def __init__(self, scenario, agent_conf, agent_obsk=0, global_categories=None, seed=None, episode_limit=1000, **kwargs):
        self.scenario = scenario  # e.g. Ant-v4
        self.agent_conf = agent_conf  # e.g. '2x3'
        self.agent_obsk = agent_obsk
        self.k_categories = self._get_local_categories(**kwargs)
        self.global_categories = self._get_global_categories(global_categories)
        self.seed = seed
        self.episode_limit = episode_limit

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario,
                                                                                             self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        self.actions_per_agent = [len(agent_partition) for agent_partition in self.agent_partitions]

        self.k_dicts = [get_joints_at_kdist(self.agent_partitions[agent_id], self.mujoco_edges, k=self.agent_obsk) 
                        for agent_id in range(self.n_agents)]

        self._create_multi_agent_env(**kwargs)
        self._create_agents_observation_spaces()
        self._create_agents_action_spaces()
        self.set_rng = True
        self.was_test_mode_before = False
        self.reset()
        self.train_rng = self.env.unwrapped.np_random


    def _get_local_categories(self, **kwargs):
        # Categories for current agent, Categories for neighbour agents (obsk > 0)
        
        if self.scenario in ["Ant-v4", "manyagent_ant"]:
            if kwargs.get('use_contact_forces', False):
                categories = ["qpos,qvel,cfrc_ext", "qpos"]
            else:
                # Gymansium.MuJoCo.Ant-v4 has disabled cfrc_ext by default
                categories = ["qpos,qvel", "qpos"]
        
        elif self.scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            categories = ["qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator", "qpos"]
        
        elif self.scenario in ["Reacher-v4"]:
            categories = ["qpos,qvel,fingertip_dist", "qpos"]
        
        elif self.scenario in ["coupled_half_cheetah"]:
            categories = ["qpos,qvel,ten_J,ten_length,ten_velocity"]
        
        else:
            categories = ["qpos,qvel", "qpos"]
        
        local_categories = [
            categories[k if k < len(categories) else -1].split(",")
            for k in range(self.agent_obsk + 1)
        ]
        return local_categories
        

    def _get_global_categories(self, global_categories):
        if global_categories is None:
            return []
        global_categories = global_categories.split(",")

        if self.scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            available_categories = ["qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"]
        else:
            available_categories = ["qpos", "qvel"]

        for category in global_categories:
            if category not in available_categories:
                raise ValueError(f"Invalid global category {category}. Available for {self.scenario} are {available_categories}")
        
        return global_categories
    

    def _create_env_custom_args(self, **kwargs):
        custom_args = kwargs.copy()
        if 'obs_entity_mode' in kwargs:
            custom_args.pop('obs_entity_mode')
        if 'state_entity_mode' in kwargs:
            custom_args.pop('state_entity_mode')
        return custom_args
    

    def _create_multi_agent_env(self, **kwargs):
        if self.scenario in ['manyagent_ant', 'manyagent_swimmer']:
            if self.scenario == 'manyagent_ant':
                env_class = ManyAgentAntEnv
            else:
                env_class = ManyAgentSwimmerEnv

            custom_args = kwargs.copy()
            custom_args['agent_conf'] = self.agent_conf
            self.env = TimeLimit(env_class(**custom_args), self.episode_limit)

        else:
            # Gym's make already applies TimeLimit in the environments like Ant-v4
            custom_args = self._create_env_custom_args(**kwargs)
            self.env = gym.make(self.scenario, **custom_args)

        self.unwrapped_env = self.env.unwrapped
        self.env.reset()
    

    def _create_agents_observation_spaces(self):
        low = np.min(self.env.observation_space.low)
        high = np.max(self.env.observation_space.high)
        self.observation_spaces = []
        for i in range(self.n_agents):
            shape = len(self.get_obs_agent(i))
            self.observation_spaces.append(
                Box(low, high, (shape,))
            )


    def _create_agents_action_spaces(self):
        low = np.min(self.env.action_space.low)
        high = np.max(self.env.action_space.high)
        self.action_spaces = tuple([
            Box(low, high, (self.actions_per_agent[a], ), seed=self.seed) 
            for a in range(self.n_agents)])
        

    def step(self, actions):
        # we need to map actions back into MuJoCo action space
        num_actions = sum([action_space.low.shape[0] for action_space in self.action_spaces])
        env_actions = np.zeros(num_actions)

        for a, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                env_actions[body_part.act_ids] = actions[a][i]

        if np.isnan(actions).any():
            raise Exception("FATAL: At least one action is NaN!")

        _, reward_n, done_n, truncated_n, info_n = self.env.step(env_actions)
        self.steps += 1

        info = {}
        info.update(info_n)

        if done_n or truncated_n:
            if self.steps < self.episode_limit:
                info["episode_limit"] = False
            else:
                info["episode_limit"] = True

        return reward_n, done_n, truncated_n, info
    

    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        obs_n = []
        for a in range(self.n_agents):
            obs_i = self.get_obs_agent(a)
            obs_n.append(obs_i)
        return obs_n


    def get_obs_agent(self, agent_id):
        return build_obs(self.env.data, self.k_dicts[agent_id], self.k_categories, 
                         self.mujoco_globals, self.global_categories)


    def get_obs_shape(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])


    def get_state(self):
        # _get_obs() returns the full-observable state
        return self.unwrapped_env._get_obs()


    def get_state_shape(self):
        return len(self.get_state())


    def get_stats(self):
        return {}


    def reset(self, test_mode=False):
        self.steps = 0

        if test_mode and not self.was_test_mode_before:
            seed = self.seed
            self.was_test_mode_before = True
            self.train_rng = self.env.unwrapped.np_random
        
        elif not test_mode and self.was_test_mode_before:
            self.was_test_mode_before = False
            seed = None
            self.env.unwrapped.np_random = self.train_rng
        else:
            seed = None

        if self.set_rng:
            seed = self.seed
            self.set_rng = False

        self.env.reset(seed=seed)
        return self.get_obs()


    def render(self):
        return self.env.render()


    def close(self):
        self.env.close()
    

    def get_action_shape(self):
        return self.action_spaces[0].shape[0]
    

    def get_action_dtype(self):
        return self.action_spaces[0].dtype
    

    def has_discrete_actions(self):
        return False
    

    def get_action_spaces(self):
        return self.action_spaces
