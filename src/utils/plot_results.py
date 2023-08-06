import matplotlib.pyplot as plt
import seaborn as sns
from utils.results_utils import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



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
    return steps, means


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
    return steps, means
