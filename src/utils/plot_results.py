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


def get_saved_steps(results_path, index, metric_name):
    metrics = load_results(results_path, index)
    steps = metrics[metric_name]['steps']
    return steps


def get_env_name(results_path, index):
    config = load_results(results_path, index, 'config.json')
    if 'key' in config['env_args']:
        env_name = config['env_args']['key']
    else:
        env_name = config['env_args']['scenario']
    env_name = env_name.split('_')[0]
    env_name = env_name[0].upper() + env_name[1:]
    return env_name


def get_algorithm_name(results_path, index):
    config = load_results(results_path, index, 'config.json')
    algorithm_name = config['name'].upper()
    return algorithm_name


def get_total_steps(results_path, index):
    config = load_results(results_path, index, 'config.json')
    return config['t_max']


def compute_mean_std_runs(results_path, start_index, end_index, metric_name):
    returns = []
    for index in range(start_index, end_index+1):
        metrics_i = load_results(results_path, index)
        returns_i = metrics_i[metric_name]['values']
        returns.append(returns_i)

    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0)
    return mean, std


def plot_test_return_over_runs(results_path, start_index, end_index):
    sns.set(style="whitegrid")
    steps = get_saved_steps(results_path, start_index, 'test_return_mean')
    total_steps = get_total_steps(results_path, start_index)
    means, stds = compute_mean_std_runs(results_path, start_index, end_index, 'test_return_mean')
    env_name = get_env_name(results_path, start_index)
    algorithm = get_algorithm_name(results_path, start_index)
    save_path = os.path.join(results_path, str(end_index))
    print(save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, means-stds, means+stds, alpha=0.3, label='Mean+-Std')
    plt.title(f'100 episodes average test return on {env_name} using {algorithm} over {total_steps} steps')
    plt.xlabel('Step')
    plt.ylabel('Test Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'episode_test_return.png'))
    plt.close()


def plot_test_episode_length_over_runs(results_path, start_index, end_index):
    sns.set(style="whitegrid")
    steps = get_saved_steps(results_path, start_index, 'test_ep_length_mean')
    total_steps = get_total_steps(results_path, start_index)
    means, stds = compute_mean_std_runs(results_path, start_index, end_index, 'test_ep_length_mean')
    env_name = get_env_name(results_path, start_index)
    algorithm = get_algorithm_name(results_path, start_index)
    save_path = os.path.join(results_path, str(end_index))
    print(save_path)

    plt.figure(figsize=(11, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, means-stds, means+stds, alpha=0.3, label='Mean+-Std')
    plt.title(f'100 episodes average episode length on {env_name} using {algorithm} over {total_steps} steps')
    plt.xlabel('Step')
    plt.ylabel('Test Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'episode_test_ep_length.png'))
    plt.close()


def plot_train_return_over_runs(results_path, start_index, end_index, logger_steps):
    sns.set(style="whitegrid")
    steps = get_saved_steps(results_path, start_index, 'return_mean')
    total_steps = get_total_steps(results_path, start_index)
    means, stds = compute_mean_std_runs(results_path, start_index, end_index, 'return_mean')
    env_name = get_env_name(results_path, start_index)
    algorithm = get_algorithm_name(results_path, start_index)
    save_path = os.path.join(results_path, str(end_index))
    print(save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Mean', color='orange')
    plt.fill_between(steps, means-stds, means+stds, alpha=0.3, label='Mean+-Std')
    plt.title(f'Last {logger_steps} steps average return on {env_name} using {algorithm} over {total_steps} steps')
    plt.xlabel('Step')
    plt.ylabel('Train Episode Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'steps_train_return.png'))
    plt.close()



if __name__ == '__main__':

    path = './results/sacred/iql/pistonball_reward_2_actions'
    start_index = 12
    end_index = 12

    plot_test_return_over_runs(path, start_index, end_index)