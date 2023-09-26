import torch as th
import time
from runners.runner import Runner
from components.episode_buffer import ReplayBuffer
from utils.timehelper import time_left, time_str
from runners.episodes_runner import EpisodesRunner



class Trainer(Runner):

    def __init__(self, config, logger, save_path):
        super().__init__(config, logger, save_path)
        self._create_replay_buffer()
        self.train_runner = EpisodesRunner(self.args, logger, self.agent_controller, self.episode_batch_function)
        self.test_runner = EpisodesRunner(self.args, logger, self.agent_controller, self.episode_batch_function, 
                                          is_test=True)

        self.num_episodes = 0
        self.last_test_step = - self.args.test_interval - 1
        self.last_save_model_step = 0
        self.last_print_stats_step = 0

        self.start_time = time.time()
        self.last_time = self.start_time


    def _create_replay_buffer(self):
        max_length = 2 if self.args.buffer_transitions else self.env_info["episode_limit"] + 1
        self.buffer = ReplayBuffer(self.scheme, self.groups, self.args.buffer_size, max_length, 
                                   preprocess=self.preprocess, device="cpu")


    def run(self):
        self.logger.console_logger.info(f'Training {self.args.env} with {self.args.name} for {self.args.t_max} timesteps')

        if self.args.checkpoint_path != '':
            step_to_load = self._load_model_checkpoint()
            #self.train_runner.total_steps = step_to_load

        while self.train_runner.total_steps <= self.args.t_max:
            
            episode_batch = self.train_runner.run_episode()

            self.num_episodes += self.train_runner.num_envs

            self._insert_batch_to_replay_buffer(episode_batch)

            if self._is_time_to_train():
                self._train_networks(episode_batch)

            if self._is_time_to_test():
                self._perform_test_episodes()
                self._print_remaining_time()
                self.last_test_step = self.train_runner.total_steps
            
            if self._is_time_to_save_models():
                self._save_model_checkpoint(self.train_runner.total_steps)
                self.last_save_model_step = self.train_runner.total_steps
            
            if self._is_time_to_print_stats():
                self.logger.log_stat("episode", self.num_episodes, self.train_runner.total_steps)
                self.logger.print_recent_stats()
                self.last_print_stats_step = self.train_runner.total_steps

        if self.args.save_model_end:
            self._save_model_checkpoint(self.train_runner.total_steps)

        self.train_runner.close_envs()
        self.test_runner.close_envs()
        self.logger.log_date_to_console()


    def _insert_batch_to_replay_buffer(self, episode_batch):
        if self.args.buffer_transitions:
            for i in range(episode_batch.batch_size):
                for t in range(episode_batch.max_seq_length - 1):
                    if t < episode_batch.max_seq_length - 1 and episode_batch['filled'][i, t] and not episode_batch['filled'][i, t+1]:
                        break
                    self.buffer.insert_episode_batch(episode_batch[i, t:t+2])

        else:
            self.buffer.insert_episode_batch(episode_batch)

    
    def _is_time_to_train(self):
        return (self.train_runner.total_steps >= self.args.start_steps and 
                self.buffer.can_sample(self.args.batch_size))
    

    def _get_num_batches_to_sample(self, episode_batch):
        n_batches_to_sample = self.args.n_batches_to_sample
        if self.args.buffer_transitions and n_batches_to_sample == -1:
            # Sample as many batches as steps performed in the environment
            n_batches_to_sample = th.sum(episode_batch['filled'][:, :episode_batch.max_seq_length - 1]).item()
        return n_batches_to_sample


    def _train_networks(self, episode_batch):
        n_batches_to_sample = self._get_num_batches_to_sample(episode_batch)   
        
        for _ in range(n_batches_to_sample):

            episode_sample = self.buffer.sample(self.args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)

            self.learner.train(episode_sample, self.train_runner.total_steps)


    def _is_time_to_test(self):
        return self.train_runner.total_steps - self.last_test_step >= self.args.test_interval
    

    def _perform_test_episodes(self):
        self.test_runner.total_steps = self.train_runner.total_steps
        self.test_runner.set_rng = True
        n_test_runs = max(1, self.args.test_nepisode // self.test_runner.num_envs)
        for i in range(n_test_runs):
            self.test_runner.run_episode()
        

    def _print_remaining_time(self):
        self.logger.console_logger.info(f't_env: {self.train_runner.total_steps} / {self.args.t_max}')
        remaining_time = time_left(self.last_time, self.last_test_step, self.train_runner.total_steps, self.args.t_max) 
        time_passed = time_str(time.time() - self.start_time)
        self.logger.console_logger.info(f'Estimated time left: {remaining_time}. Time passed: {time_passed}')
        self.last_time = time.time()


    def _is_time_to_save_models(self):
        return (self.args.save_model and 
                self.train_runner.total_steps - self.last_save_model_step >= self.args.save_model_interval)
    

    def _is_time_to_print_stats(self):
        return self.train_runner.total_steps - self.last_print_stats_step >= self.args.log_interval