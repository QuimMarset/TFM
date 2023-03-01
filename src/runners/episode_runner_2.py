from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner2:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=self.args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.last_learn_T = 0


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, 
            self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess


    def get_env_info(self):
        return self.env.get_env_info()


    def save_replay(self):
        self.env.save_replay()


    def close_env(self):
        self.env.close()


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()

        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        self.batch.update(pre_transition_data, ts=0)
        self.t = 0


    def run(self, test_mode=False, **kwargs):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            
            actions = self.mac.select_actions(self.batch, t_ep=self.t, 
                t_env=self.t_env, test_mode=test_mode)

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, ts=self.t)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, ts=self.t)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, ts=self.t)

            if False and (not test_mode) and (getattr(self.args, "runner_scope", "episodic") == "transition"):
                buffer = kwargs.get("buffer")
                learner = kwargs.get("learner")
                episode = kwargs.get("episode")

                # insert single transitions into buffer
                # note zeros inserted for batch elements already terminated
                # buffer.insert_episode_batch(self.batch[:, self.t-1:self.t+1])
                buffer.insert_episode_batch(self.batch[0, self.t - 1:self.t + 1])

                if (self.t_env + self.t - self.last_learn_T) / self.args.learn_interval >= 1.0:
                    # execute learning steps (if enabled)
                    if buffer.can_sample(self.args.batch_size) and (buffer.episodes_in_buffer > getattr(self.args, "buffer_warmup", 0)):
                        episode_sample = buffer.sample(self.args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != self.args.device:
                            episode_sample.to(self.args.device)

                        if self.args.verbose:
                            print("Learning now for {} steps...".format(getattr(self.args, "n_train", 1)))
                        for _ in range(getattr(self.args, "n_train", 1)):
                            learner.train(episode_sample, self.t_env, episode)
                        self.last_learn_T = self.t_env + self.t

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size

        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)

        elif (not test_mode) and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac, "action_selector") and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        self.mac.ou_noise_state = actions.clone().zero_()

        return self.batch


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()