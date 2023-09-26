import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.results_utils import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def get_manyagent_swimmer_return_y_ticks(algorithm_name, agent_conf):
    if agent_conf == '2x2':
        if algorithm_name in ['td3', 'ddpg']:
            return range(-150, 501, 50)
        else:
            return range(-150, 301, 50)
    elif agent_conf == '4x2':
        return range(-150, 501, 50)
    elif agent_conf == '10x2':
        return range(-350, 601, 50)
    else:
        raise ValueError(f'Ticks not defined for {agent_conf}')


def get_ant_return_y_ticks(algorithm_name, agent_conf):
    return range(-1500, 5500, 500)


def get_episode_return_plot_y_ticks(env_name, algorithm_name, agent_conf=None):
    if env_name == 'Swimmer-v4':
        return range(-150, 401, 50)
    elif env_name == 'manyagent_swimmer' or env_name == 'ManySegmentSwimmer':
        return get_manyagent_swimmer_return_y_ticks(algorithm_name, agent_conf)
    elif env_name == 'pistonball':
        return range(-50, 120, 10)
    elif env_name == 'Ant-v4':
        return get_ant_return_y_ticks(algorithm_name, agent_conf)
    else:
        raise ValueError(f'Unknown environment {env_name}')


def get_episode_length_plot_y_ticks(env_name):
    if env_name in ['Swimmer-v4', 'Ant-v4', 'manyagent_swimmer', 'ManySegmentSwimmer']:
        return None
    elif env_name == 'pistonball':
        return range(0, 126, 25)
    else:
        raise ValueError(f'Unknown environment {env_name}')


def stylize_algorithm_name(algorithm_name):
    if algorithm_name in ['qmix', 'facmac', 'maddpg', 'td3', 'ddpg', 'dqn', 'iql', 'vdn']:
        return algorithm_name.upper()
    elif algorithm_name == 'facmac_td3':
        return 'FACMAC-TD3'
    elif algorithm_name == 'maddpg_discrete':
        return 'MADDPG (Gumbel-Softmax)'
    elif algorithm_name == 'transf_qmix_discrete':
        return 'Transf-QMIX'
    elif algorithm_name == 'iql_continuous':
        return 'IQL (Continuous)'
    else:
        raise ValueError(f'Unknown algorithm name {algorithm_name}')
    

def stylize_environment_name(env_name, agent_conf=None):
    if env_name == 'pistonball':
        return 'Pistonball'
    elif env_name in ['Swimmer-v4', 'Ant-v4']:
        return env_name
    elif env_name == 'manyagent_swimmer':
        return f'ManyAgent Swimmer {agent_conf}'
    elif env_name == 'ManySegmentSwimmer':
        return f'ManySegment Swimmer {agent_conf}'
    else:
        raise ValueError(f'Unknown environment name {env_name}')


def plot_metric_values_over_runs(folder_path, num_repetitions, metric_name, y_label, plot_title, y_ticks_range):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    steps = get_min_steps(folder_path, num_repetitions, metric_name)
    mean_values = compute_mean_value_over_runs(folder_path, num_repetitions, metric_name, len(steps))

    for i in range(1, num_repetitions + 1):
        values_i = load_results(folder_path, i)[metric_name]['values'][:len(steps)]
        plt.plot(steps, values_i, color=sns.color_palette("mako")[i], label=f'Run {i}', alpha=0.6)

    plt.plot(steps, mean_values, color='red', label='Mean')
    
    if y_ticks_range is not None:
        plt.yticks(y_ticks_range)
    
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('Step')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'plot_{metric_name}.png'), dpi=300)
    plt.close()


def plot_test_return_over_runs(folder_path, num_repetitions, env_name, algorithm_name):
    config = load_config(folder_path)
    agent_conf = config['env_args'].get('agent_conf', None)
    y_ticks_range = get_episode_return_plot_y_ticks(env_name, algorithm_name, agent_conf)
    
    y_label = 'Test Episode Return'
    algorithm_name = stylize_algorithm_name(algorithm_name)
    env_name = stylize_environment_name(env_name, agent_conf)
    plot_title = f'Test episode return on {env_name} using {algorithm_name}'
    
    plot_metric_values_over_runs(folder_path, num_repetitions, 'test_return_mean', y_label, 
                                 plot_title, y_ticks_range)
    

def plot_train_return_over_runs(folder_path, num_repetitions, env_name, algorithm_name):
    config = load_config(folder_path)
    agent_conf = config['env_args'].get('agent_conf', None)
    y_ticks_range = get_episode_return_plot_y_ticks(env_name, algorithm_name, agent_conf)
    
    y_label = 'Train Episode Return'
    algorithm_name = stylize_algorithm_name(algorithm_name)
    env_name = stylize_environment_name(env_name, agent_conf)
    plot_title = f'Train episode return on {env_name} using {algorithm_name}'
    
    plot_metric_values_over_runs(folder_path, num_repetitions, 'return_mean', y_label, 
                                 plot_title, y_ticks_range)


def plot_test_episode_length_over_runs(folder_path, num_repetitions, env_name, algorithm_name):
    config = load_config(folder_path)
    agent_conf = config['env_args'].get('agent_conf', None)
    y_ticks_range = get_episode_length_plot_y_ticks(env_name)
    
    y_label = 'Test Episode Length'
    algorithm_name = stylize_algorithm_name(algorithm_name)
    env_name = stylize_environment_name(env_name, agent_conf)
    plot_title = f'Test episode length on {env_name} using {algorithm_name}'
    
    plot_metric_values_over_runs(folder_path, num_repetitions, 'test_ep_length_mean', y_label, 
                                 plot_title, y_ticks_range)
