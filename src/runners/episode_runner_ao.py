import torch as th
from copy import deepcopy
import numpy as np
from functools import partial
from envs import REGISTRY as env_REGISTRY
from components.episode_buffer import EpisodeBatch
from envs.adaptive_optics.delayed_mdp import DelayedMDP
from components.transition_buffer import TransitionsReplayBuffer



class EpisodeRunnerAdaptiveOptics:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.last_log_train_step = -1000000

        self.delay = self.args.env_args.get('delayed_assignment', 1)
        self.delayed_mdp = DelayedMDP(self.delay)


    def setup(self, scheme, groups, preprocess, mac):
        self.buffer = TransitionsReplayBuffer(self.args.buffer_size, deepcopy(scheme), groups)
        self.mac = mac

        if 'delayed_assignment' in self.args.env_args:
            reward_delay = self.args.env_args['delayed_assignment'] + 2
        else:
            reward_delay = 1
        
        self.new_batch = partial(EpisodeBatch, scheme, groups, 1, self.episode_limit + reward_delay,
                                 preprocess=preprocess, device=self.args.device)


    def get_env_info(self):
        return self.env.get_env_info(self.args)


    def save_replay(self):
        pass


    def close_env(self):
        self.env.close()


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

        pre_transition_data = {
            "state": [self.env.get_state()],
            "obs": [self.env.get_obs()]
        }

        self.batch.update(pre_transition_data, ts=self.t)
        self.delayed_mdp = DelayedMDP(self.delay)
      

    def run(self, test_mode=False, **kwargs):
        learner = kwargs.get('learner', None)

        self.reset()

        state = self.env.get_state()
        obs = self.env.get_obs()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=1)

        while not terminated:

            action = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            action = action.detach()
            cpu_action = action.to("cpu").numpy()

            reward, done, truncated, env_info = self.env.step(cpu_action[0])
            terminated = done or truncated

            next_obs = self.env.get_obs()
            next_state = self.env.get_state()
                
            episode_return += reward

            if not test_mode and self.delayed_mdp.check_update_possibility():
                state_delayed, obs_delayed, action_delayed, next_state_delayed, next_obs_delayed = self.delayed_mdp.credit_assignment()

                data = {
                    'state': [state_delayed],
                    'obs': [obs_delayed],
                    'actions': [action_delayed], 
                    'reward': [reward],
                    'terminated': [done],
                    'next_state': [next_state_delayed],
                    'next_obs': [next_obs_delayed],
                }

                self.buffer.add_transitions(data)

            if not test_mode:
                self.delayed_mdp.save(state, obs, cpu_action[0], next_state, next_obs)

            post_transition_data = {
                "actions": action,
                "reward": [(reward,)],
                "terminated": [(done,)],
            }           
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            if not test_mode:
                self.t_env += 1

            state = next_state
            obs = next_obs

            pre_transition_data = {
                "state": [self.env.get_state()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            if not test_mode and self.t_env >= self.args.start_steps and self.buffer.can_sample(self.args.batch_size):
                train_data = self.buffer.sample(self.args.batch_size)
                if train_data.device != self.args.device:
                    train_data.to(self.args.device)

                learner.train(train_data, self.t_env)
           

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        episode_return /= self.episode_limit

        cur_returns.append(episode_return)
        self.logger.write_episode_return(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        
        elif self.t_env - self.last_log_train_step >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            
            self.last_log_train_step = self.t_env

        return None


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
