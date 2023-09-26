from envs import REGISTRY as env_REGISTRY
import os
from PIL import Image
import numpy as np
import torch as th



class EpisodesRunner:

    def __init__(self, args, logger, agent_controller, episode_batch_creator, is_test=False):
        self.args = args
        self.logger = logger
        self.num_envs = self.args.num_envs
        self.agent_controller = agent_controller
        self.episode_batch_creator = episode_batch_creator
        self.is_test = is_test

        self._create_envs()

        self.set_rng = True
        self.total_steps = 0
        self.last_log_stats_step = - self.args.runner_log_interval - 1
        self.returns = []
        self.stats = {}

        
    def _create_envs(self):
        env_fn = env_REGISTRY[self.args.env]
        self.env_seeds = []
        
        env_args = [self.args.env_args.copy() for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            env_args[i]["seed"] = self.args.env_args['seed'] + i

            if self.is_test and env_args[i].get('scenario', '') == 'Ant-v4':
                env_args[i]['healthy_reward'] = 1

            self.env_seeds.append(env_args[i]["seed"])

        self.envs = [env_fn(**env_args_i) for env_args_i in env_args]
        

    def close_envs(self):
        for env in self.envs:
            env.close()


    def reset(self):
        self.step = 0
        # executed_steps counts all the steps executed in all environments
        # compared to total_steps that considers one environment (i.e. sequential steps)
        self.executed_steps = 0
        self.batch = self.episode_batch_creator()

        pre_transition_data = {
            "state": [],
            "obs": []
        }

        for i, env in enumerate(self.envs):

            """ 
                Set the environment RNG when required:
                    If are training and we have multiple envs -> Reset it every time
                    If we are training and we have one env -> Reset it once
                    If we are testing -> Reset it once
                We reset it every time in one case as when we train with multiple envs, we use enough to 
                generate enough randomness and avoid overfitting those particular episodes.
                Moreover, the RNG only changes the initial state, and in large-duration episodes it does not 
                affect very much
            """

            if self.set_rng or self.num_envs > 1 and not self.is_test:
                seed = self.env_seeds[i]
            else:
                seed = None

            env.reset(seed)
            pre_transition_data["state"].append(env.get_state())
            pre_transition_data["obs"].append(env.get_obs())

        self.set_rng = False

        self.batch.update(pre_transition_data, ts=0)

        
    def run_episode(self, experiment_path=None, episode_num=None):
        self.reset()
        self.agent_controller.init_hidden(batch_size=self.num_envs)

        all_terminated = False
        episode_returns = [0 for _ in range(self.num_envs)]
        episode_lengths = [0 for _ in range(self.num_envs)]
        terminated = [False for _ in range(self.num_envs)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        while True:

            if self._is_time_to_save_frames():
                self._save_episode_frames(experiment_path, episode_num, self.step)
            
            actions = self.agent_controller.select_actions(self.batch, t_ep=self.step, t_env=self.total_steps, 
                                                           bs=envs_not_terminated, test_mode=self.is_test)
            actions = actions.detach()
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.step, mark_filled=False)

            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            if getattr(self.args, "num_previous_transitions", 0) > 0:
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

                    post_transition_data["reward"].append((reward,))

                    episode_returns[idx] += reward
                    episode_lengths[idx] += 1
                    
                    self.executed_steps += 1
                    if not self.is_test and getattr(self.args, 'increase_step_counter', True):
                        self.total_steps += 1

                    if done or truncated:
                        final_env_infos.append(env_info)

                    terminated[idx] = done or truncated
                    post_transition_data["terminated"].append((done,))

                    if getattr(self.args, "num_previous_transitions", 0) > 0:
                        post_transition_data['prev_obs'].append(self._build_prev_obs(idx, self.step))
                        post_transition_data['prev_actions'].append(self._build_prev_actions(idx, self.step))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(env.get_state())
                    pre_transition_data["obs"].append(env.get_obs())

                    action_idx += 1

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.step, mark_filled=False)

            # Move onto the next timestep
            self.step += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.step)

        if not self.is_test and not getattr(self.args, 'increase_step_counter', True):
            self.total_steps += self.executed_steps

        self._update_stats(final_env_infos, episode_lengths)
        self.returns.extend(episode_returns)

        for episode_return in episode_returns:
            self.logger.write_episode_return(episode_return, self.is_test)

        if self._is_time_to_log():
            prefix = 'test_' if self.is_test else ''
            self._log_returns(prefix)
            self._log_stats(prefix)
            self.last_log_stats_step = self.total_steps

        return self.batch
    

    def _update_stats(self, episode_stats, episode_lengths):
        for episode_stats_i in episode_stats:
            for key in episode_stats_i:
                self.stats[key] = self.stats.get(key, 0) + episode_stats_i.get(key, 0)

        self.stats["n_episodes"] = self.num_envs + self.stats.get("n_episodes", 0)
        self.stats["ep_length"] = sum(episode_lengths) + self.stats.get("ep_length", 0)

    
    def _is_time_to_log(self):
        n_test_runs = max(1, self.args.test_nepisode // self.num_envs) * self.num_envs

        if self.is_test and len(self.returns) == n_test_runs:
            return True
        elif not self.is_test and self.total_steps - self.last_log_stats_step >= self.args.runner_log_interval:
            return True
        return False
        
    
    def _log_returns(self, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(self.returns), self.total_steps)
        self.logger.log_stat(prefix + "return_std", np.std(self.returns), self.total_steps)
        self.returns.clear()

    
    def _log_stats(self, prefix):
        for k, v in self.stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v / self.stats["n_episodes"], self.total_steps)

        if (not self.is_test and hasattr(self.agent_controller, "action_selector") 
            and hasattr(self.agent_controller.action_selector, "epsilon")):
            self.logger.log_stat("epsilon", self.agent_controller.action_selector.epsilon, self.total_steps)
            
        self.stats.clear()


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
        #TODO: Fix actions  in discrete case
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
    

    def _is_time_to_save_frames(self):
        is_evaluate_time = getattr(self.args, 'evaluate', False)
        save_frames = getattr(self.args, 'save_frames', False)
        return is_evaluate_time and save_frames


    def _save_episode_frames(self, experiment_path, episode_num, frame_num):
        folder_path = os.path.join(experiment_path, 'frames', f'frames_{episode_num}')
        os.makedirs(folder_path, exist_ok=True)
        
        frame = self.envs[0].render()
        if frame is not None:
            image = Image.fromarray(frame)
            frame_path = os.path.join(folder_path, f'frame_{frame_num}.png')
            image.save(frame_path)