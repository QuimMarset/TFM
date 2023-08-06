import sys
from copy import deepcopy
from multiprocessing import Process
from run import run
from utils.plot_results import (plot_train_return_over_runs, plot_test_return_over_runs, 
                                plot_test_episode_length_over_runs)
from utils.main_utils import *
from utils.logging_results import Logger, GlobalTensorboardLogger



def set_repetition_seeds(config, repetition_index):
    if 'seed' not in config:
        seed = get_random_seed()
        config['seed'] = seed
    else:
        config['seed'] += repetition_index * config['batch_size_run']
    config['env_args']['seed'] = config['seed']
    set_random_seed(config['seed'])


def run_repetition(config, experiment_path, repetition_index):
    set_repetition_seeds(config, repetition_index)
    logger = Logger(experiment_path, repetition_index)
    logger.save_config(config)
    run(config, logger, experiment_path)



if __name__ == '__main__':

    default_env_config_name = 'mujoco_multi'
    default_alg_config_name = 'facmac'

    params = deepcopy(sys.argv)
    params_dict = input_args_to_dict(params)

    env_config_name = params_dict.get('env_config_name', default_env_config_name)
    algorithm_config_name = params_dict.get('algorithm_config_name', default_alg_config_name)

    env_config = get_env_cofig(env_config_name)
    alg_config = get_algorithm_config(algorithm_config_name)
    default_config = get_default_config()

    config = update_config_recursive(default_config, env_config)
    config = update_config_recursive(config, alg_config)

    if params_dict:
        config = update_config_recursive(config, params_dict)

    env_name = get_env_name(config)
    algorithm_name = get_algorithm_name(config)
    
    experiment_path = create_new_experiment(env_name, algorithm_name)

    processes = []
    num_repetitions = config['repetitions']
    for index in range(1, num_repetitions + 1):
        run_path = create_new_experiment_run(experiment_path, index)
        process = Process(target=run_repetition, args=(config.copy(), run_path, index - 1,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    if not config['evaluate']:
        train_steps, train_mean_returns = plot_train_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)
        test_steps, test_mean_returns = plot_test_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)
        plot_test_episode_length_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)

        global_logger = GlobalTensorboardLogger(experiment_path)
        global_logger.log_global_stats(num_repetitions)