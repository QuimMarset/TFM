import numpy as np
import os
from functools import partial
import torch as th
from types import SimpleNamespace as SN
from components.episode_buffer import EpisodeBatch
from learners import REGISTRY as le_REGISTRY
from envs import REGISTRY as env_REGISTRY
from components.transforms import OneHot
from utils.run_utils import *



class Runner:


    def __init__(self, config, logger, save_path):
        self.config = self._sanity_check(config, logger.console_logger)
        self.logger = logger
        self.save_path = save_path

        self.logger.print_config(config)
        
        self.args = SN(**config)
        self.args.device = 'cuda' if self.args.use_cuda else 'cpu'

        self._get_env_info()
        self._fill_args_with_env_info()
        # Scheme, preprocessing, and groups are used to create the EpisodeBatch and ReplayBuffer 
        self._create_scheme()
        self._create_preprocessing_ops()
        self.groups = {"agents": self.args.n_agents}
        self._create_episode_batch_creator()
        self._create_learner()
        self._create_agent_controller()


    def _get_env_info(self):
        env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env_info = env.get_env_info(self.args)
        env.close()


    def _fill_args_with_env_info(self):
        for k, v in self.env_info.items():
            setattr(self.args, k, v)

        self.args.n_net_outputs = get_agent_network_num_of_outputs(self.args, self.env_info)
        # If the method is not an actor-critic, this second variable is not used
        self.args.n_critic_net_outputs = get_critic_network_num_of_outputs(self.args, self.env_info)


    def _create_scheme(self):
        self.actions_dtype = th.float if self.env_info['action_dtype'] == np.float32 else th.long

        self.scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": self.env_info['action_shape'], "group": "agents", "dtype": self.actions_dtype},
            "reward": {"vshape": 1},
            "terminated": {"vshape": 1, "dtype": th.uint8},
        }

        if (self.args.buffer_transitions and getattr(self.args, 'num_previous_transitions', 0) > 0 
            and is_multi_agent_method(self.args)):

            self.scheme['prev_obs'] = {
                'vshape': self.env_info['obs_shape'] * self.args.num_previous_transitions,
                'group' : 'agents'
            }
            self.scheme['prev_actions'] = {
                "vshape": self.env_info['action_shape'] * self.args.num_previous_transitions, 
                "group": "agents", 
                "dtype": self.actions_dtype
            }

    
    def _create_preprocessing_ops(self):
        if self.actions_dtype == th.float:
            self.preprocess = {}
        else:
            self.preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_discrete_actions)])
            }


    def _create_episode_batch_creator(self):
        self.episode_limit = self.args.episode_limit
        self.episode_batch_function = partial(EpisodeBatch, self.scheme, 
                                              self.groups, self.args.num_envs, self.episode_limit + 1,
                                              preprocess=self.preprocess, device=self.args.device)
        self.episode_batch = self.episode_batch_function()


    def _create_learner(self):
        # Scheme is modified depending on the preprocessing operations (only applies to discrete actions)
        self.learner = le_REGISTRY[self.args.learner](self.episode_batch.scheme, self.logger, self.args)
        if self.args.use_cuda:
            self.learner.cuda()

    
    def _create_agent_controller(self):
        if getattr(self.learner, 'actor', None) is None:
            self.agent_controller = self.learner.agent
        else:
            self.agent_controller = self.learner.actor


    def _get_model_checkpoint_saved_timesteps(self):
        timesteps = []
        
        for saved_timestep in os.listdir(self.args.checkpoint_path):
            path = os.path.join(self.args.checkpoint_path, saved_timestep)
            if os.path.isdir(path) and saved_timestep.isdigit():
                timesteps.append(int(saved_timestep))

        return timesteps
    

    def _load_model_checkpoint(self):
        timesteps = self._get_model_checkpoint_saved_timesteps()

        if self.args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))

        model_path = os.path.join(self.args.checkpoint_path, str(timestep_to_load))
        self.logger.console_logger.info("Loading model from {}".format(model_path))
        self.learner.load_models(model_path)
        return timestep_to_load
    

    def _save_model_checkpoint(self, timestep):
        save_path_model = os.path.join(self.save_path, 'models', str(timestep))
        os.makedirs(save_path_model, exist_ok=True)
        self.logger.console_logger.info("Saving models to {}".format(save_path_model))
        self.learner.save_models(save_path_model)

    
    def _sanity_check(self, config, logger):
        if config["use_cuda"] and not th.cuda.is_available():
            config["use_cuda"] = False
            logger.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

        if config["test_nepisode"] < config["num_envs"]:
            config["test_nepisode"] = config["num_envs"]
        else:
            config["test_nepisode"] = (config["test_nepisode"]//config["num_envs"]) * config["num_envs"]

        if config['env'] == 'adaptive_optics':
            config['add_agent_id'] = False
            config['critic_add_agent_id'] = False
            
            if 'num_previous_transitions' in config:
                config['num_previous_transitions'] = 0

            if 'add_last_action' in config:
                config['add_last_action'] = False
                config['critic_add_last_action'] = False

            if config['name'] in ['td3', 'ddpg']:
                config['env_args']['partition'] = 1

        return config
