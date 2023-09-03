from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th



class ParallelRunnerOneThread:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        env_fn = env_REGISTRY[self.args.env]
        
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] = self.args.env_args['seed'] + i

        self.envs = [env_fn(**env_args_i) for env_args_i in env_args]

        self.env_info = self.envs[0].get_env_info(self.args)
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0
        self.log_train_stats_t = -100000

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess


    def get_env_info(self):
        return self.env_info


    def save_replay(self):
        pass


    def close_env(self):
        for env in self.envs:
            env.close()


    def reset(self, test_mode):
        self.batch = self.new_batch()

        pre_transition_data = {
            "state": [],
            "obs": []
        }

        for env in self.envs:
            env.reset(test_mode)
            pre_transition_data["state"].append(env.get_state())
            pre_transition_data["obs"].append(env.get_obs())

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.executed_steps = 0


    def run(self, test_mode=False, **kwargs):
        self.reset(test_mode)
        self.mac.init_hidden(batch_size=self.batch_size)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                                              bs=envs_not_terminated, test_mode=test_mode)
            actions = actions.detach()
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            if getattr(self.args, "num_previous_transitions", -1) > 0:
                post_transition_data["prev_obs"] = []
                post_transition_data['prev_actions'] = []
            
            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break
            
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "obs": []
            }

            action_idx = 0
            for idx, env in enumerate(self.envs):
                if idx in envs_not_terminated:
                    
                    reward, done, truncated, env_info = env.step(cpu_actions[action_idx])

                    if isinstance(reward, (list, tuple)):
                        assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                        reward = reward[0]

                    post_transition_data["reward"].append((reward,))

                    episode_returns[idx] += reward
                    episode_lengths[idx] += 1
                    
                    if not test_mode:
                        self.executed_steps += 1
                        if getattr(self.args, 'increase_step_counter', True):
                            self.t_env += 1

                    if done or truncated:
                        final_env_infos.append(env_info)

                    terminated[idx] = done or truncated
                    post_transition_data["terminated"].append((done,))

                    if getattr(self.args, "num_previous_transitions", -1) > 0:
                        post_transition_data['prev_obs'].append(self._build_prev_obs(idx, self.t))
                        post_transition_data['prev_actions'].append(self._build_prev_actions(idx, self.t))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(env.get_state())
                    pre_transition_data["obs"].append(env.get_obs())

                    action_idx += 1

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t)

        if not test_mode and not getattr(self.args, 'increase_step_counter', True):
            self.t_env += self.executed_steps

        # Get stats back for each env
        env_stats = []
        for env in self.envs:
            env_stats.append(env.get_stats())

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        for episode_return in episode_returns:
            self.logger.write_episode_return(episode_return)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac, "action_selector") and hasattr(self.mac.action_selector, "epsilon"):
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


    def _build_prev_obs(self, batch_index, t):
        if t >= self.args.num_previous_transitions:
            obs = [self.batch["obs"][batch_index, i] for i in range(t-self.args.num_previous_transitions, t)]
        
        else:
            obs = [self.batch["obs"][batch_index, i] for i in range(t)]
            for _ in range(t, self.args.num_previous_transitions):
                obs.append(self.batch['obs'][batch_index, t])
        
        obs = th.cat(obs, dim=-1).cpu().numpy()
        return obs


    def _build_prev_actions(self, batch_index, t):
        window = self.args.num_previous_transitions
        
        if t == 0:
            actions = [th.zeros_like(self.batch["actions"][batch_index, t]) for _ in range(window)]
        
        elif t >= window:
            actions = [self.batch["actions"][batch_index, i] for i in range(t-window, t)]
        
        else:
            actions = [self.batch["actions"][batch_index, i] for i in range(t)]
            for _ in range(t, window):
                actions.append(self.batch['actions'][batch_index, t-1])
        
        actions = th.cat(actions, dim=-1).detach().cpu().numpy()
        return actions
    