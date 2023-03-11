from os.path import join, isdir
from os import makedirs, listdir
import collections
import yaml
import numpy as np
import torch
from utils.constant_paths import *



def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def get_env_cofig(env_name):
    yaml_path = join(env_configs_path, f'{env_name}.yaml')
    return read_yaml(yaml_path)


def get_algorithm_config(alg_name):
    yaml_path = join(alg_configs_path, f'{alg_name}.yaml')
    return read_yaml(yaml_path)


def get_default_config():
    return read_yaml(default_config_path)


def update_config_recursive(default_config, loaded_config):
    for key, value in loaded_config.items():
        if isinstance(value, collections.Mapping):
            default_config[key] = update_config_recursive(default_config.get(key, {}), value)
        else:
            default_config[key] = value
    return default_config


def get_experiments_folder(env_name, algorithm_name):
    path = join(results_path, env_name, algorithm_name)
    makedirs(path, exist_ok=True)
    return path


def get_number_subfolders(path):
    return sum(isdir(join(path, elem)) for elem in listdir(path))


def create_new_experiment(experiments_folder_path):
    num_experiments = get_number_subfolders(experiments_folder_path)
    experiment_path = join(experiments_folder_path, str(num_experiments + 1))
    makedirs(experiment_path, exist_ok=True)
    return experiment_path


def set_random_seed():
    seed = np.random.randint(0, 99999)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def set_correct_type(string_value):
    if string_value.isnumeric():
        return int(string_value)
    elif '.' in string_value and string_value.replace('.', '').isnumeric():
        return float(string_value)
    elif string_value == 'True':
        return True
    elif string_value == 'False':
        return False
    return string_value


def input_args_to_dict(input_args):
    args_dict = {}
    for arg in input_args[1:]:
        key, value = arg.split("=")
        if len(key.split("--")) > 1:
            key = key.split('--')[1]
        args_dict[key] = set_correct_type(value)
    return args_dict