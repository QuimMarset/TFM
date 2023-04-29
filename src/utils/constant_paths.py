from os.path import join
from os import makedirs


root_path = ''
src_path = join(root_path, '')

configs_path = join(src_path, 'config')
alg_configs_path = join(configs_path, 'algs')
env_configs_path = join(configs_path, 'envs')
default_config_path = join(configs_path, 'default.yaml')


results_path = join(root_path, 'results')
makedirs(results_path, exist_ok=True)
