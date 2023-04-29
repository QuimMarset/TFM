import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def load_results(results_path, index, file_name='metrics.json'):
    path = os.path.join(results_path, str(index), file_name)
    with open(path, 'r') as file:
        metrics = json.load(file)
    return metrics


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


def plot_test_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name):
    sns.set(style="whitegrid")
    steps = get_saved_steps(experiment_path, num_repetitions, 'test_return_mean')
    #means, stds = compute_mean_std_runs(experiment_path, num_repetitions, 'test_return_mean')
    means, mins, maxs = compute_mean_min_max_runs(experiment_path, num_repetitions, 'test_return_mean')
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, mins, maxs, alpha=0.3, label='[Min, Max]')
    plt.title(f'Average test return on {env_name} using {algorithm_name}')
    plt.xlabel('Step')
    plt.ylabel('Test Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'episode_test_return.png'))
    plt.close()


def plot_test_episode_length_over_runs(experiment_path, num_repetitions, env_name, algorithm_name):
    sns.set(style="whitegrid")
    steps = get_saved_steps(experiment_path, num_repetitions, 'test_ep_length_mean')
    #means, stds = compute_mean_std_runs(experiment_path, num_repetitions, 'test_ep_length_mean')
    means, mins, maxs = compute_mean_min_max_runs(experiment_path, num_repetitions, 'test_ep_length_mean')
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, mins, maxs, alpha=0.3, label='[Min, Max]')
    plt.title(f'Average test episode length on {env_name} using {algorithm_name}')
    plt.xlabel('Step')
    plt.ylabel('Test Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'episode_test_ep_length.png'))
    plt.close()


def plot_train_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name):
    sns.set(style="whitegrid")
    steps = get_saved_steps(experiment_path, num_repetitions, 'return_mean')
    #means, stds = compute_mean_std_runs(experiment_path, num_repetitions, 'return_mean')
    means, mins, maxs = compute_mean_min_max_runs(experiment_path, num_repetitions, 'return_mean')
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, mins, maxs, alpha=0.3, label='[Min, Max]')
    plt.title(f'Average train episode return on {env_name} using {algorithm_name}')
    plt.xlabel('Step')
    plt.ylabel('Train Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'steps_train_return.png'))
    plt.close()
