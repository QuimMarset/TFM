from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th
import numpy as np
from PIL import Image
import os



class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1, 'Episode runner runs a single environment'

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # Used in the Adaptive Optics environment as reward comes with frame delay
        if 'delayed_assignment' in self.args.env_args:
            reward_delay = self.args.env_args['delayed_assignment'] + 2
        else:
            reward_delay = 1
        
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + reward_delay,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

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


    def run(self, test_mode=False, **kwargs):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            if getattr(self.args, 'evluate', False) and getattr(self.args, 'save_frames', False) and kwargs.get('episode_num', 0) == 0:
                self._save_episode_frames(kwargs.get('experiment_path'), kwargs.get('episode_num'), self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = actions.detach()
            cpu_actions = actions.to("cpu").numpy()

            reward, done, truncated, env_info = self.env.step(cpu_actions[0])
            if self.args.env in ["particle"] and isinstance(reward, (list, tuple)):
                reward = reward[0]

            terminated = done or truncated
                
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(done,)],
            }
            if getattr(self.args, "num_previous_transitions", -1) > 0:
                post_transition_data["prev_obs"] = self._build_prev_obs(self.t)
                post_transition_data['prev_actions'] = self._build_prev_actions(self.t)
            
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            if not test_mode:
                self.t_env += 1

            pre_transition_data = {
                "state": [self.env.get_state()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)
        self.logger.write_episode_return(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


    def _build_prev_obs(self, t):
        if t >= self.args.num_previous_transitions:
            obs = [self.batch["obs"][:, i] for i in range(t-self.args.num_previous_transitions, t)]
        
        else:
            obs = [self.batch["obs"][:, i] for i in range(t)]
            for _ in range(t, self.args.num_previous_transitions):
                obs.append(self.batch['obs'][:, t])
        
        obs = th.cat(obs, dim=-1)
        return obs


    def _build_prev_actions(self, t):
        window = self.args.num_previous_transitions
        
        if t == 0:
            actions = [th.zeros_like(self.batch["actions"][:, t]) for _ in range(window)]
        
        elif t >= window:
            actions = [self.batch["actions"][:, i] for i in range(t-window, t)]
        
        else:
            actions = [self.batch["actions"][:, i] for i in range(t)]
            for _ in range(t, window):
                actions.append(self.batch['actions'][:, t-1])
        
        actions = th.cat(actions, dim=-1)
        return actions
    

    def _save_episode_frames(self, experiment_path, episode_num, frame_num):
        folder_path = os.path.join(experiment_path, 'frames', f'frames_{episode_num}')
        os.makedirs(folder_path, exist_ok=True)
        
        frame = self.env.render()
        if frame is not None:
            image = Image.fromarray(frame)
            frame_path = os.path.join(folder_path, f'frame_{frame_num}.png')
            image.save(frame_path)