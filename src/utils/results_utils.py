import os
import numpy as np
import json



def load_results(folder_path, index, file_name='metrics.json'):
    path = os.path.join(folder_path, 'run_' + str(index), file_name)
    with open(path, 'r') as file:
        metrics = json.load(file)
    return metrics


def load_config(folder_path):
    path = os.path.join(folder_path, 'run_1', 'config.json')
    with open(path, 'r') as file:
        config = json.load(file)
    return config


def get_metric_names(results_path):
    metrics = load_results(results_path, 1)
    return list(metrics.keys())


def get_min_steps(folder_path, num_repetitions, metric_name):
    min_num_steps = None
    steps = []

    for num_repetition in range(1, num_repetitions + 1):

        metrics_repetition = load_results(folder_path, num_repetition)
        steps_metric_repetition = metrics_repetition[metric_name]['steps']

        if min_num_steps is None or len(steps_metric_repetition) < min_num_steps:
            min_num_steps = len(steps_metric_repetition)
            steps = steps_metric_repetition
    
    return steps


def compute_mean_value_over_runs(folder_path, num_repetitions, metric_name, num_steps):
    metric_values = []
    
    for index in range(1, num_repetitions + 1):
        metrics_repetition = load_results(folder_path, index)
        metric_values_i = metrics_repetition[metric_name]['values']
        metric_values.append(metric_values_i[:num_steps])
    
    mean = np.mean(metric_values, axis=0)
    return mean