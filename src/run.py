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
    if args.save_replay:
        runner.save_replay()
    runner.close_env()


def run_sequential(args, logger, save_path):
    
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.action_shape = env_info['action_shape']
    args.n_discrete_actions = env_info['n_discrete_actions']
    args.action_spaces = env_info.get('action_spaces', [])

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
        "actions": {"vshape": (env_info['action_shape'], ), "group": "agents", "dtype": actions_dtype},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
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

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, 
                          env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)

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

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            
            for _ in range(args.n_batches_to_sample):

                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env)
                
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

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

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
