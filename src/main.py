import sys
from copy import deepcopy
from run import run
from utils.plot_results import *
from utils.main_utils import *
from utils.logging_results import Logger



def run_repetition(config, experiment_path):
    seed = set_random_seed()
    config['seed'] = seed
    logger = Logger(experiment_path)
    logger.save_config(config)
    run(config, logger, experiment_path)



if __name__ == '__main__':

    env_name = 'pettingzoo_continuous'
    algorithm_name = 'facmac_pettingzoo'

    params = deepcopy(sys.argv)
    params_dict = input_args_to_dict(params)

    env_config = get_env_cofig(env_name)
    alg_config = get_algorithm_config(algorithm_name)
    default_config = get_default_config()

    config = update_config_recursive(default_config, env_config)
    config = update_config_recursive(config, alg_config)

    if params_dict:
        config = update_config_recursive(config, params_dict)

    if 'key' in config['env_args']:
        env_name = config['env_args']['key']
    else:
        env_name = config['env_args']['scenario']
    
    experiments_path = get_experiments_folder(env_name, algorithm_name)
    start_index = get_number_subfolders(experiments_path) + 1

    num_repetitions = config['repetitions']
    for i in range(num_repetitions):
        experiment_path = create_new_experiment(experiments_path)
        run_repetition(config, experiment_path)

    end_index = get_number_subfolders(experiments_path)

    if not config['evaluate']:
        plot_train_return_over_runs(experiments_path, start_index, end_index, config['runner_log_interval'])
        plot_test_return_over_runs(experiments_path, start_index, end_index)
        plot_test_episode_length_over_runs(experiments_path, start_index, end_index)
