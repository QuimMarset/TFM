import os
from os.path import dirname, abspath
import collections
import torch as th
import yaml
from experiment import ex, set_config, set_observer
from utils.plot_results import plot_test_return_over_runs, get_last_results_index, plot_train_return_over_runs



def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



if __name__ == '__main__':

    params = ['src/main.py', '--config=qmix', '--env-config=pettingzoo']

    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]
    
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]
    
    results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name.replace(':', '_')}")

    set_config(config_dict)
    start_index = get_last_results_index(file_obs_path) + 1

    for i in range(config_dict['repetitions']):
        set_observer(file_obs_path)
        ex.run_commandline(params)

    end_index = get_last_results_index(file_obs_path)
    plot_train_return_over_runs(file_obs_path, start_index, end_index, config_dict['runner_log_interval'])
    plot_test_return_over_runs(file_obs_path, start_index, end_index)


"""
p, sobre, dreta, esq
Incrementar time_penalty
Indepenents laterals
Energia del pistó
Epsilon deixar més temps al mínim perquè aprengui
Comparar 2 versions (indep, comaprtit) amb un DQN
"""
# IQL reward a tots el mateix
# 5 10 per cada mètode