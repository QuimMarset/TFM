from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th



class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] = self.args.env_args['seed'] + i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", self.args))
        self.env_info = self.parent_conns[0].recv()
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
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "obs": []
        }
        # Get the obs and state back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, **kwargs):
        self.reset()
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

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            if getattr(self.args, "num_previous_transitions", -1) > 0:
                post_transition_data["prev_obs"] = []
                post_transition_data['prev_actions'] = []
            
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    if data["done"] or data['truncated']:
                        final_env_infos.append(data["info"])

                    terminated[idx] = data["done"] or data['truncated']
                    post_transition_data["terminated"].append((data['done'],))

                    if getattr(self.args, "num_previous_transitions", -1) > 0:
                        post_transition_data['prev_obs'].append(self._build_prev_obs(idx, self.t))
                        post_transition_data['prev_actions'].append(self._build_prev_actions(idx, self.t))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

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



def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, done, truncated, env_info = env.step(actions)
            if isinstance(reward, (list, tuple)):
                assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                reward = reward[0]
            # Return the observations and state to make the next action
            state = env.get_state()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "done" : done,
                "truncated": truncated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            args = data
            remote.send(env.get_env_info(args))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
