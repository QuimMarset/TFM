

# n_agents refers to the number of agent the environment assumes it has

def get_agent_network_num_of_outputs(args, env_info):

    if env_info['has_discrete_actions']:
        
        if args.name == 'dqn':
            return env_info['n_discrete_actions'] ** args.n_agents
        
        else:
            return env_info['n_discrete_actions']
    
    else:

        agent_types = ['rnn_mean_log_std', 'rnn_actor', 'mlp_actor', 'transformer_actor', 'mlp_mean_log_std',
                   'non_shared_actor_mlp', 'non_shared_actor_rnn', 'sides_actor_rnn', 'sides_actor_mlp',
                   'shared_but_first_actor_mlp', 'cnn_actor', 'mlp_actor_ao']

        if args.name == 'td3' or args.name == 'ddpg':
            return env_info['action_shape'] * args.n_agents

        elif args.agent in agent_types:
            return env_info['action_shape']

        else:
            # COMIX
            return 1



def get_critic_network_num_of_outputs(args, env_info):
    return 1


def is_multi_agent_method(args):
    return args.name not in ['td3', 'dqn', 'ddpg']