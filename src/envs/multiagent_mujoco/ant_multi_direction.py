import gym
from gym.spaces import Box
import mujoco
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.multiagent_mujoco.obsk import get_joints_at_kdist, get_parts_and_edges, build_obs



class AntMultiDirectionMultiAgentEnv(MultiAgentEnv):

    def __init__(self, agent_conf, agent_obsk=0, global_categories=None, seed=None, episode_limit=1000, **kwargs):
        self.scenario = 'Ant-v4'
        self.agent_conf = agent_conf  # e.g. '2x3'
        self.agent_obsk = agent_obsk
        self.k_categories = self._get_local_categories(**kwargs)
        self.global_categories = self._get_global_categories(global_categories)
        self.seed = seed
        self.episode_limit = episode_limit
        self.use_contact_forces = kwargs.get('use_contact_forces', False)
        self.num_resets = 0

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario,
                                                                                             self.agent_conf)
        self.original_agent_partitions = self.agent_partitions.copy()
        self._create_parts_dict()

        self.n_agents = len(self.agent_partitions)
        assert self.n_agents > 1, "This environment only admits 2 or 4 agents, not 1"

        self.actions_per_agent = [len(agent_partition) for agent_partition in self.agent_partitions]

        self._create_ant_env(**kwargs)
        self.reset()
        self._create_agents_observation_spaces()
        self._create_agents_action_spaces()


    def _get_local_categories(self, **kwargs):
        # Categories for current agent, Categories for neighbour agents (obsk > 0)
        
        if kwargs.get('use_contact_forces', False):
            categories = ["qpos,qvel,cfrc_ext", "qpos"]
        else:
            # Gymansium.MuJoCo.Ant-v4 has disabled cfrc_ext by default
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

        available_categories = ["qpos", "qvel"]

        for category in global_categories:
            if category not in available_categories:
                raise ValueError(f"Invalid global category {category}. Available for {self.scenario} are {available_categories}")
        
        return global_categories
    

    def _create_parts_dict(self):
        # hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4
        parts, _, _ = get_parts_and_edges(self.scenario, None)
        parts = parts[0]
        self.parts = {
            'hip_1' : parts[0],
            'ankle_1' : parts[1],
            'hip_2' : parts[2],
            'ankle_2' : parts[3],
            'hip_3' : parts[4],
            'ankle_3' : parts[5],
            'hip_4' : parts[6],
            'ankle_4' : parts[7]
        }
    

    def _create_env_custom_args(self, **kwargs):
        custom_args = kwargs.copy()
        if 'obs_entity_mode' in kwargs:
            custom_args.pop('obs_entity_mode')
        if 'state_entity_mode' in kwargs:
            custom_args.pop('state_entity_mode')
        if 'scenario' in kwargs:
            custom_args.pop('scenario')
        return custom_args
    

    def _create_ant_env(self, **kwargs):
        custom_args = self._create_env_custom_args(**kwargs)
        self.env = gym.make(self.scenario, **custom_args)
        self.unwrapped_env = self.env.unwrapped
    

    def _create_agents_observation_spaces(self):
        low = np.min(self.env.observation_space.low)
        high = np.max(self.env.observation_space.high)
        self.observation_spaces = []
        for i in range(self.n_agents):
            shape = len(self.get_obs_agent(i))
            self.observation_spaces.append(Box(low, high, (shape,)))


    def _create_agents_action_spaces(self):
        low = np.min(self.env.action_space.low)
        high = np.max(self.env.action_space.high)
        self.action_spaces = tuple([
            Box(low, high, (self.actions_per_agent[a], ), seed=self.seed) 
            for a in range(self.n_agents)])
        

    def step(self, actions, is_test=False):
        # we need to map actions back into MuJoCo action space
        num_actions = sum([action_space.low.shape[0] for action_space in self.action_spaces])
        env_actions = np.zeros(num_actions)

        for a, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                env_actions[body_part.act_ids] = actions[a][i]

        _, reward_n, done_n, truncated_n, info_n = self.env.step(env_actions)
        self.steps += 1

        if is_test and self.random_direction == 0:
            reward_n *= -1

        info = {}
        info.update(info_n)

        if done_n or truncated_n:
            if self.steps < self.episode_limit:
                info["episode_limit"] = False
            else:
                info["episode_limit"] = True

        return reward_n, done_n, truncated_n, info
    

    def _select_random_direction(self):
        self.random_direction = self.num_resets % 2
        self.num_resets += 1
    

    def _change_legs_order_two_agents(self, is_test=False):
        if is_test:
            return [
                (self.parts['hip_1'], self.parts['ankle_1'], self.parts['hip_2'], self.parts['ankle_2']),
                (self.parts['hip_4'], self.parts['ankle_4'], self.parts['hip_3'], self.parts['ankle_3'])
            ]

        if self.random_direction == 0:
            # left
            agent_partitions = [
                (self.parts['hip_3'], self.parts['ankle_3'], self.parts['hip_4'], self.parts['ankle_4']),
                (self.parts['hip_2'], self.parts['ankle_2'], self.parts['hip_1'], self.parts['ankle_1'])
            ]
        else:
            # right
            agent_partitions = [
                (self.parts['hip_1'], self.parts['ankle_1'], self.parts['hip_2'], self.parts['ankle_2']),
                (self.parts['hip_4'], self.parts['ankle_4'], self.parts['hip_3'], self.parts['ankle_3'])
            ]
        return agent_partitions
    

    def _change_legs_order(self, is_test=False):
        if self.n_agents == 2:
            self.agent_partitions = self._change_legs_order_two_agents(is_test)
        else:
            raise NotImplemented('Pending to implement the leg order change for 4 agents')
        
        self.k_dicts = [get_joints_at_kdist(self.agent_partitions[agent_id], self.mujoco_edges, k=self.agent_obsk) 
                        for agent_id in range(self.n_agents)]
        
    
    def get_obs(self, is_test=False):
        obs_n = []
        for a in range(self.n_agents):
            obs_i = self.get_obs_agent(a, is_test)
            obs_n.append(obs_i)
        return obs_n


    def get_obs_agent(self, agent_id, is_test=False):
        obs = build_obs(self.env.data, self.k_dicts[agent_id], self.k_categories, 
                         self.mujoco_globals, self.global_categories)

        if self.random_direction == 0 and not is_test:
            obs[8:10] *= -1
            obs[11:13] *= -1
        
        return np.concatenate([obs, [self.random_direction]])


    def get_obs_shape(self):
        return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])
    

    def rotate_quaternion_180_z(self, original_quaternion):
        rot_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rot_quat, np.array([0, 0, 1]), np.deg2rad(180))
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, original_quaternion, rot_quat)
        return new_quat
        

    def get_state(self, is_test=False):
        agents_data = []
        for agent_dict in self.k_dicts:
            for agent_part in agent_dict[0]:
                pos = self.env.data.qpos[agent_part.qpos_ids]
                vel = self.env.data.qvel[agent_part.qvel_ids]
                agents_data.extend([pos, vel])

        torso_data = []
        torso = self.mujoco_globals[0]
        for field_name in torso.extra_obs:
            if not self.use_contact_forces and field_name == 'cfrc_ext':
                continue
            torso_data.extend(torso.extra_obs[field_name](self.env.data).tolist())

        torso_data = np.array(torso_data)

        if self.random_direction == 0 and not is_test:
            torso_data[1:5] = self.rotate_quaternion_180_z(torso_data[1:5])
            torso_data[5:7] *= -1
            torso_data[8:10] *= -1
            pass

        state = np.concatenate([agents_data, torso_data, [self.random_direction]])
        return state


    def get_state_shape(self):
        return len(self.get_state())


    def get_stats(self):
        return {}


    def reset(self, is_test=False, seed=None):
        self.steps = 0
        self._select_random_direction()
        self._change_legs_order(is_test)
        self.env.reset(seed=seed)
        return self.get_obs(is_test)


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
