import os
import numpy as np
import json


def load_results(results_path, index, file_name='metrics.json'):
    path = os.path.join(results_path, 'run_' + str(index), file_name)
    with open(path, 'r') as file:
        metrics = json.load(file)
    return metrics


def get_metric_names(results_path):
    metrics = load_results(results_path, 1)
    return list(metrics.keys())


def get_saved_steps(results_path, num_repetitions, metric_name):
    min_len, index = get_min_length(results_path, num_repetitions, metric_name)
    metrics = load_results(results_path, index)
    steps = metrics[metric_name]['steps'][:min_len]
    return steps


def get_min_length(results_path, num_repetitions, metric_name):
    min_len = None
    index_min_len = -1
    for index in range(1, num_repetitions+1):
        metrics_i = load_results(results_path, index)
        steps_metric_i = metrics_i[metric_name]['steps']
        if min_len is None or len(steps_metric_i) < min_len:
            min_len = len(steps_metric_i)
            index_min_len = index
    return min_len, index_min_len


def compute_mean_std_runs(results_path, num_repetitions, metric_name):
    returns = []
    min_len, _ = get_min_length(results_path, num_repetitions, metric_name)
    for index in range(1, num_repetitions+1):
        metrics_i = load_results(results_path, index)
        returns_i = metrics_i[metric_name]['values']
        returns.append(returns_i[:min_len])
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0)
    return mean, std


def compute_mean_min_max_runs(results_path, num_repetitions, metric_name):
    metric_values = []
    min_len, _ = get_min_length(results_path, num_repetitions, metric_name)
    for index in range(1, num_repetitions+1):
        metrics_repetition = load_results(results_path, index)
        metric_values_i = metrics_repetition[metric_name]['values']
        metric_values.append(metric_values_i[:min_len])
    mean = np.mean(metric_values, axis=0)
    min = np.min(metric_values, axis=0)
    max = np.max(metric_values, axis=0)
    return mean, min, max


def compute_mean_runs(results_path, num_repetitions, metric_name):
    values = []
    min_len, _ = get_min_length(results_path, num_repetitions, metric_name)
    for index in range(1, num_repetitions+1):
        metrics_i = load_results(results_path, index)
        values_i = metrics_i[metric_name]['values']
        values.append(values_i[:min_len])
    return np.mean(values, axis=0)