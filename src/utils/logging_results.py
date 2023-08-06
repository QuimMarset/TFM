from collections import defaultdict
import logging
import numpy as np
import os
import json
import pprint
from datetime import datetime, date
from tensorboard_logger import Logger as TBLogger
from utils.results_utils import *



class Logger:

    def __init__(self, save_path, repetition_index):
        self.console_logger = self.create_console_logger(repetition_index)
        self.stats = defaultdict(lambda: [])
        self.save_path = save_path
        self.metrics_path = os.path.join(save_path, 'metrics.json')
        self.log_output_path = os.path.join(save_path, 'console_otput.txt')
        self.returns_path = os.path.join(save_path, 'episode_returns.txt')
        self.episode_num = 0
        self.create_metrics_data()
        self.create_tensorboard_logger()
        self.log_date_to_console()


    def create_metrics_data(self):
        self.write_metrics_data({})


    def create_tensorboard_logger(self):
        self.tensorboard_logger = TBLogger(self.save_path)


    def read_metrics_data(self):
        with open(self.metrics_path, 'r') as file:
            return json.load(file)


    def write_metrics_data(self, data):
        with open(self.metrics_path, 'w+') as file:
            json.dump(data, file, indent=4)
        

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        metrics_data = self.read_metrics_data()

        self.tensorboard_logger.log_value(key, value, t)
        
        if key not in metrics_data:
            metrics_data[key] = {
                'steps' : [],
                'values' : []
            }
        metrics_data[key]['steps'].append(t)
        metrics_data[key]['values'].append(value)

        self.write_metrics_data(metrics_data)


    def write_console_output(self, string_data):
        with open(self.log_output_path, 'a') as file:
            file.write(string_data + '\n')


    def write_episode_return(self, episode_return):
        with open(self.returns_path, 'a') as file:
            file.write(f'Episode {self.episode_num} return: {episode_return:.4f}\n')
        self.episode_num += 1


    def print_recent_stats(self):
        log_str = "\nRecent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
        self.write_console_output(log_str)


    def save_config(self, config):
        config_path = os.path.join(self.save_path, 'config.json')
        with open(config_path, 'w+') as file:
            json.dump(config, file, indent=4)

    
    def print_config(self, config):
        self.console_logger.info("Experiment Parameters:")
        experiment_params = pprint.pformat(config, indent=4, width=1)
        self.console_logger.info("\n\n" + experiment_params + "\n")


    def create_console_logger(self, repetition_index):
        logger = logging.getLogger(f'Run_{repetition_index + 1}')
        logger.handlers = []
        
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.setLevel('DEBUG')
        return logger
    

    def log_date_to_console(self):
        current_date = date.today().strftime("%d/%m/%Y")
        current_time = datetime.now().strftime("%H:%M:%S")
        date_and_time = f'Current date is {current_date} and current time is {current_time}'
        self.write_console_output(date_and_time)


class GlobalTensorboardLogger:

    def __init__(self, experiment_path):
        self.log_stats = []
        self.experiment_path = experiment_path
        self.configure_global_tensorboard(experiment_path)

    def configure_global_tensorboard(self, experiment_path):
        experiment_tb_path = os.path.join(experiment_path, 'average_tb_logs')
        os.makedirs(experiment_tb_path, exist_ok=True)
        self.tensorboard_logger = TBLogger(experiment_tb_path)


    def log_global_stats_metric(self, metric_name, global_stats, steps):
        for global_stat, step in zip(global_stats, steps):
            self.tensorboard_logger.log_value(metric_name, global_stat, step)


    def log_global_stats(self, num_repetitions):
        metric_names = get_metric_names(self.experiment_path)
        for metric_name in metric_names:
            steps = get_min_steps(self.experiment_path, num_repetitions, metric_name)
            mean_stats = compute_mean_value_over_runs(self.experiment_path, num_repetitions, metric_name, len(steps))
            self.log_global_stats_metric(metric_name, mean_stats, steps)
