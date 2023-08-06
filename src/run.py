import numpy as np
import os
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.timehelper import time_left, time_str
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.run_utils import *



def run(config, logger, save_path):

    # check args sanity
    config = args_sanity_check(config, logger.console_logger)

    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger.print_config(config)

    # Run and train
    run_sequential(args, logger, save_path)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)
    if getattr(args, 'save_replay', False):
        runner.save_replay()
    runner.close_env()


def run_sequential(args, logger, save_path):
    
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    args.n_net_outputs = get_agent_network_num_of_outputs(args, env_info)
    # If the method is not an actor-critic, this second variable is not used
    args.n_critic_net_outputs = get_critic_network_num_of_outputs(args, env_info)

    if env_info['action_dtype'] == np.float32:
        actions_dtype = th.float
    else:
        actions_dtype = th.long

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": env_info['action_shape'], "group": "agents", "dtype": actions_dtype},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    if args.buffer_transitions and getattr(args, 'num_previous_transitions', 0) > 0 and is_multi_agent_method(args):
        scheme['prev_obs'] = {
            'vshape': env_info['obs_shape'] * args.num_previous_transitions,
            'group' : 'agents'
        }
        scheme['prev_actions'] = {
            "vshape": env_info['action_shape'] * args.num_previous_transitions, 
            "group": "agents", 
            "dtype": actions_dtype
        }

    groups = {
        "agents": args.n_agents
    }

    if args.env_args.get('continuous', True) or actions_dtype == th.float:
        preprocess = {}
    else:
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_discrete_actions)])
        }

    max_length = 2 if args.buffer_transitions else env_info["episode_limit"] + 1
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, max_length, preprocess=preprocess, device="cpu")

    # Learner
    learner = le_REGISTRY[args.learner](buffer.scheme, logger, args)
    # Setup multiagent controller here
    if getattr(learner, 'actor', None) is None:
        mac = learner.agent
    else:
        mac = learner.actor

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        #timestep_to_load = 1360000

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = - args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        episode_batch = runner.run(test_mode=False, **{'learner': learner})

        if args.runner != 'episode_ao':

            if args.buffer_transitions:
                if args.env == 'adaptive_optics':
                    update_replay_buffer_adaptive_optics(episode_batch, args, buffer)
                else:
                    for i in range(episode_batch.batch_size):
                        for t in range(episode_batch.max_seq_length):
                            buffer.insert_episode_batch(episode_batch[i, t:t+2])
            else:
                buffer.insert_episode_batch(episode_batch)
            
            if runner.t_env >= args.start_steps and buffer.can_sample(args.batch_size):
                
                n_batches_to_sample = args.n_batches_to_sample
                if args.buffer_transitions and n_batches_to_sample == -1:
                    # Sample as many batches as steps performed in the environment
                    reward_delay = args.env_args.get('delayed_assignment', 1)
                    n_batches_to_sample = th.sum(episode_batch['filled'][:, :episode_batch.max_seq_length - reward_delay]).item()
                
                for _ in range(n_batches_to_sample):

                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env)
                
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if runner.t_env >= args.start_steps and runner.t_env - last_test_T >= args.test_interval:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path_model = os.path.join(save_path, 'models', str(runner.t_env))
            os.makedirs(save_path_model, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path_model))
            learner.save_models(save_path_model)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if args.save_model_end:
        save_path_model = os.path.join(save_path, 'models', str(runner.t_env))
        os.makedirs(save_path_model, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path_model))
        learner.save_models(save_path_model)

    runner.close_env()
    logger.console_logger.info("Finished Training")
    logger.log_date_to_console()


def update_replay_buffer_adaptive_optics(episode_batch, args, buffer):
    delay = args.env_args['delayed_assignment'] + 2

    for i in range(episode_batch.batch_size):
        for t in range(episode_batch.max_seq_length - delay):
            
            reward = [(episode_batch[i, t + delay]['reward'].item(),)]
            episode_batch.update({'reward' : reward}, ts=t)
            buffer.insert_episode_batch(episode_batch[i, [t, t + delay]])
            

def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

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
