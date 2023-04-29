

# n_agents refers to the number of agent the environment assumes it has

def get_agent_network_num_of_outputs(args, env_info):

    if env_info['has_discrete_actions']:
        
        if args.name == 'dqn':
            return env_info['n_discrete_actions'] ** args.n_agents
        
        else:
            return env_info['n_discrete_actions']
    
    else:

        if args.agent == 'rnn_mean_log_std' or args.agent == 'rnn_actor' or args.agent == 'mlp_actor':
            return env_info['action_shape']
        
        elif args.name == 'single_actor_critic':
            return env_info['action_shape'] * args.n_agents

        else:
            # COMIX
            return 1



def get_critic_network_num_of_outputs(args, env_info):
    return 1