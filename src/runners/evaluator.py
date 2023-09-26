import os
from PIL import Image
from runners.runner import Runner
from runners.episodes_runner import EpisodesRunner



class Evaluator(Runner):


    def __init__(self, config, logger, save_path):
        super().__init__(config, logger, save_path)
        self.test_episodes = self.args.test_nepisode
        self.runner = EpisodesRunner(self.args, logger, self.agent_controller, 
                                     self.episode_batch_function, is_test=True)  


    def run(self):
        self.logger.console_logger.info(f'Evaluating {self.args.env} with {self.args.name}')

        if self.args.checkpoint_path != '':
            step_to_load = self._load_model_checkpoint()
            self.runner.total_steps = step_to_load

        for i in range(self.test_episodes):
            self.runner.run_episode(self.save_path, i)

        self.logger.log_stat("episode", self.runner.total_steps, self.args.test_nepisode)
        self.logger.print_recent_stats()

        self.runner.close_envs()
        

    def _save_episode_frames(self, experiment_path, episode_num, frame_num):
        folder_path = os.path.join(experiment_path, 'frames', f'frames_{episode_num}')
        os.makedirs(folder_path, exist_ok=True)
        
        frame = self.env.render()
        if frame is not None:
            image = Image.fromarray(frame)
            frame_path = os.path.join(folder_path, f'frame_{frame_num}.png')
            image.save(frame_path)
