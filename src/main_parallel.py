import sys
from copy import deepcopy
from multiprocessing import Process
from runners.trainer import Trainer
from runners.evaluator import Evaluator
from utils.plot_results import *
from utils.main_utils import *
from utils.logging_results import Logger, GlobalTensorboardLogger



def set_repetition_seeds(config, repetition_index, seed):
    if 'seed' not in config:
        config['seed'] = seed
    else:
        if isinstance(config['seed'], int):
            config['seed'] += repetition_index * config['num_envs']
        elif isinstance(config['seed'], list):
            config['seed'] = config['seed'][repetition_index]
        else:
            raise ValueError(f'Only integer or list of seeds. Invalid seed {config["seed"]}')
    
    config['env_args']['seed'] = config['seed']
    set_random_seed(config['seed'])


def run_repetition(config, experiment_path, repetition_index, seed):
    set_repetition_seeds(config, repetition_index, seed)
    logger = Logger(experiment_path, repetition_index)
    logger.save_config(config)

    if config['evaluate']:
        runner = Evaluator(config, logger, experiment_path)
    else:
        runner = Trainer(config, logger, experiment_path)

    runner.run()



if __name__ == '__main__':

    default_env_config_name = 'mujoco_multi'
    default_alg_config_name = 'td3'

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

    if not config['evaluate']:
        num_repetitions = config['repetitions']
        processes = []
        
        num_repetitions = config['repetitions']
        for index in range(1, num_repetitions + 1):
            run_path = create_new_experiment_run(experiment_path, index)
            seed = get_random_seed()
            process = Process(target=run_repetition, args=(config.copy(), run_path, index - 1, seed))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        plot_train_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)
        plot_test_return_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)
        plot_test_episode_length_over_runs(experiment_path, num_repetitions, env_name, algorithm_name)

        global_logger = GlobalTensorboardLogger(experiment_path)
        global_logger.log_global_stats(num_repetitions)

    else:
        run_path = create_new_experiment_run(experiment_path, num_run=1)
        run_repetition(config.copy(), run_path, repetition_index=0, seed=get_random_seed())
