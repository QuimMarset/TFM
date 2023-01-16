import numpy as np
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
import torch as th
from utils.logging_results import get_logger
from run import run



SETTINGS['CAPTURE_MODE'] = "no"
logger = get_logger("epymarl")
ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds



def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    # run the framework
    run(_run, config, _log)


def set_config(config):
    ex.add_config(config)


def set_observer(save_path):
    ex.observers.clear()
    ex.observers.append(FileStorageObserver.create(save_path))