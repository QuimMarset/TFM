import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.multiagent_mujoco.manyagent_ant import ManyAgentAntEnv
from envs.multiagent_mujoco.manyagent_swimmer import ManyAgentSwimmerEnv
from envs.multiagent_mujoco.swimmer_v4_revised import SwimmerEnv
from envs.multiagent_mujoco.obsk import get_joints_at_kdist, get_parts_and_edges, build_obs



# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):

    def __init__(self, **kwargs):
        self.scenario = kwargs["scenario"]  # e.g. Ant-v4
        self.agent_conf = kwargs["agent_conf"]  # e.g. '2x3'

        self.seed_ = kwargs.get('seed', None)

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario,
                                                                                             self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])

        self.agent_obsk = kwargs.get("agent_obsk", None) # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs.get("agent_obsk_agents", False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs.get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v4", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                elif self.scenario in ["Reacher-v4"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"

            k_split = self.k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk+1)]

            self.global_categories_label = kwargs.get("global_categories")
            self.global_categories = self.global_categories_label.split(",") if self.global_categories_label is not None else []


        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(self.agent_partitions[agent_id],
                                                self.mujoco_edges,
                                                k=self.agent_obsk) 
                                                for agent_id in range(self.n_agents)]

        # load scenario from script
        self.episode_limit = kwargs.get('episode_limit', 1000)

        if self.scenario in ['manyagent_ant', 'manyagent_swimmer', 'Swimmer-v4']:
            if self.scenario == 'manyagent_ant':
                env_class = ManyAgentAntEnv
            elif self.scenario == 'manyagent_swimmer':
                env_class = ManyAgentSwimmerEnv
            else:
                env_class = SwimmerEnv

            self.wrapped_env = NormalizedActions(TimeLimit(env_class(**kwargs), max_episode_steps=self.episode_limit))

        else:
            custom_args = kwargs.get('custom_args', {})
            self.wrapped_env = NormalizedActions(gym.make(self.scenario, **custom_args))

        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()

        low = np.min(self.wrapped_env.observation_space.low)
        high = np.max(self.wrapped_env.observation_space.high)
        self.observation_space = [Box(low=np.array([low]*self.n_agents),
                                      high=np.array([high]*self.n_agents)) for _ in range(self.n_agents)]

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple([
            Box(self.env.action_space.low[sum(acdims[:a]):sum(acdims[:a+1])],
                self.env.action_space.high[sum(acdims[:a]):sum(acdims[:a+1])],
                seed=self.seed_) 
            for a in range(self.n_agents)])
        
            
        self.state_entity_mode = kwargs.get('state_entity_mode', False)
        if self.state_entity_mode:
            self._set_entity_attributes()
        

    def _set_entity_attributes(self):
        self.n_entities = self.env.unwrapped.n_entities_obs
        self.n_entities_obs = self.env.unwrapped.n_entities_obs
        self.n_entities_state = self.env.unwrapped.n_entities_state
        self.obs_entity_feats = self.env.unwrapped.obs_entity_feats
        self.state_entity_feats = self.env.unwrapped.state_entity_feats
        

    def step(self, actions):
        # we need to map actions back into MuJoCo action space
        num_actions = sum([action_space_i.low.shape[0] for action_space_i in self.action_space])
        env_actions = np.zeros(num_actions)

        for a, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                env_actions[body_part.act_ids] = actions[a][i]

        if np.isnan(actions).any():
            raise Exception("FATAL: At least one action is NaN!")

        obs_n, reward_n, done_n, truncated_n, info_n = self.wrapped_env.step(env_actions)
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
        if self.agent_obsk is None:
            return self.env.unwrapped._get_obs()
        else:
            obs = build_obs(self.env.data, self.k_dicts[agent_id], self.k_categories, 
                                     self.mujoco_globals, self.global_categories)
            return obs


    def get_obs_shape(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])


    def get_state(self):
        state = self.env.unwrapped._get_obs()
        return state


    def get_state_shape(self):
        """ Returns the shape of the state"""
        return len(self.get_state())


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset(seed=self.seed_)
        return self.get_obs()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
    
    def get_action_shape(self):
        return self.action_space[0].shape[0]
    
    def get_action_dtype(self):
        return self.action_space[0].dtype
    
    def has_discrete_actions(self):
        return False
    
    def get_action_spaces(self):
        return self.action_space
