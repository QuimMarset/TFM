from collections import defaultdict
import logging
import numpy as np
import os
import json
import pprint



class Logger:

    def __init__(self, save_path):
        self.console_logger = self.create_console_logger()
        self.stats = defaultdict(lambda: [])
        self.save_path = save_path
        self.metrics_path = os.path.join(save_path, 'metrics.json')
        self.create_metrics_data()


    def create_metrics_data(self):
        self.write_metrics_data({})


    def read_metrics_data(self):
        with open(self.metrics_path, 'r') as file:
            return json.load(file)


    def write_metrics_data(self, data):
        with open(self.metrics_path, 'w+') as file:
            json.dump(data, file, indent=4)
        

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        metrics_data = self.read_metrics_data()
        
        if key not in metrics_data:
            metrics_data[key] = {
                'steps' : [],
                'values' : []
            }
        metrics_data[key]['steps'].append(t)
        metrics_data[key]['values'].append(value)

        self.write_metrics_data(metrics_data)


    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
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


    def save_config(self, config):
        config_path = os.path.join(self.save_path, 'config.json')
        with open(config_path, 'w+') as file:
            json.dump(config, file, indent=4)

    
    def print_config(self, config):
        self.console_logger.info("Experiment Parameters:")
        experiment_params = pprint.pformat(config, indent=4, width=1)
        self.console_logger.info("\n\n" + experiment_params + "\n")


    def create_console_logger(self):
        logger = logging.getLogger('epymarl')
        logger.handlers = []
        
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.setLevel('DEBUG')
        return logger

