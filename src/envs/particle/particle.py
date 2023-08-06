from collections import namedtuple
from math import sqrt
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.particle.environment import MultiAgentEnv as OpenAIMultiAgentEnv
from envs.particle import scenarios
from gym import spaces


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class Particle(MultiAgentEnv):

    def __init__(self, **kwargs):

        self.args = convert(kwargs)

        # load scenario from script
        self.episode_limit = self.args.episode_limit
        self.scenario = scenarios.load(self.args.scenario + ".py").Scenario(self.args.obs_entity_mode)
        if not self.args.partial_obs:
            self.world = self.scenario.make_world()
        else:
            self.world = self.scenario.make_world(self.args)
        self.n_agents = len(self.world.policy_agents)
        self.steps = 0
        self.truncate_episodes = getattr(self.args, "truncate_episodes", True) #by default
        self.total_steps = 0

        if self.args.benchmark:
            self.env = OpenAIMultiAgentEnv(self.world,
                                            self.scenario.reset_world,
                                            self.scenario.reward,
                                            self.scenario.observation,
                                            self.scenario.benchmark_data)
        else:
            if not self.args.partial_obs:
                self.env = OpenAIMultiAgentEnv(self.world,
                                                self.scenario.reset_world,
                                                self.scenario.reward,
                                                self.scenario.observation)
            else:
                self.env = OpenAIMultiAgentEnv(self.world,
                                               self.scenario.reset_world,
                                               self.scenario.reward,
                                               self.scenario.observation,
                                               self.scenario.observation,
                                               state_callback=self.scenario.state)

        self.glob_args = kwargs.get("args")
        self._set_entity_attributes()

    
    def _set_entity_attributes(self):
        self.n_entities_obs = self.scenario.n_entities_obs
        self.obs_entity_feats = self.scenario.obs_entity_feats
        self.n_entities_state = self.scenario.n_entities_state
        self.state_entity_feats = self.scenario.state_entity_feats
        self.n_entities = self.scenario.n_entities


    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)

        # Terminate if episode_limit is reached
        self.steps += 1
        self.is_done = all(done_n)
        terminated = all(done_n)

        if self.steps >= self.episode_limit and not terminated:
            terminated = True
            info_n["episode_limit"] = getattr(self, "truncate_episodes", True)  # by default True
        else:
            info_n["episode_limit"] = False

        # test minimum distance to a landmark
        min_dists = []
        for agent in self.world.agents:
            min_dists.append(float("inf"))
            for landmark in self.world.landmarks:
                dist = sqrt(sum((apos-lpos)**2 for apos, lpos in zip(agent.state.p_pos, landmark.state.p_pos)))
                if dist < min_dists[-1]:
                    min_dists[-1] = dist

        info_n["min_dists_mean"] = np.mean(min_dists)
        if hasattr(self.scenario, "n_last_collisions"):
            info_n["n_last_collisions"] = self.scenario.n_last_collisions

        if "n" in info_n:
            del info_n["n"]

        for i, min_dist in enumerate(min_dists):
            info_n["mind_dist__agent{}".format(i)] = min_dist
        return reward_n, terminated, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i, _ in enumerate(self.world.policy_agents):
            obs = self.get_obs_agent(i)
            obs_n.append(obs)
        return obs_n

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs = self.env._get_obs(self.world.policy_agents[agent_id])
        if len(obs) < self.get_obs_shape():
            obs = np.concatenate([obs, np.zeros((self.get_obs_shape() - len(obs)))],
                                 axis=0)  # pad all obs to same length
        return obs

    def get_obs_shape(self):
        """ Returns the shape of the observation """
        return max([o.shape[0] for o in self.env.observation_space])
    
    def get_full_obs_agent(self, agent_id):
        obs = self.env._get_full_obs(self.world.policy_agents[agent_id])
        return obs

    def get_state(self, team=None):
        obs_n = []
        for i, _ in enumerate(self.world.policy_agents):
            obs = self.get_full_obs_agent(i)
            obs_n.append(obs)
        state = np.concatenate(obs_n)
        #return self.env._get_state()
        return state

    def get_state_shape(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_state())
        return state_size

    def get_number_of_possible_actions(self):
        """ Returns the total number of actions an agent could ever take """
        if all([isinstance(act_space, spaces.Discrete) for act_space in self.env.action_space]):
            return max([x.n for x in self.env.action_space])
        elif all([isinstance(act_space, spaces.Box) for act_space in self.env.action_space]):
            if self.args.scenario == "simple_speaker_listener":
                return self.env.action_space[0].shape[0] + self.env.action_space[1].shape[0]
            else:
                return max([x.shape[0] for x in self.env.action_space])
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.env.action_space]):
            return max([x.spaces[0].shape[0] + x.spaces[1].shape[0] for x in self.env.action_space])
        else:
            raise Exception("not implemented for this scenario!")

    def get_stats(self):
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self, force_reset=False):
        """ Returns initial observations and states"""
        self.total_steps += self.steps
        self.steps = 0
        if (not getattr(self.glob_args, "continuous_episode", False)) or force_reset or self.is_done:
            self.is_done = False
            self.env.reset()
        pass

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def get_action_shape(self):
        return self.env.action_space[0].shape[0]

    def close(self):
        self.env.close()

    def seed(self):
        raise NotImplementedError

    def has_discrete_actions(self):
        return False

    def get_action_dtype(self):
        return np.float32
    
    def get_action_spaces(self):
        return self.env.action_space
