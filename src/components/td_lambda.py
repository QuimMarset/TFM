import torch as th



def build_td_lambda_targets(rewards, terminated, mask, target_qs, gamma, td_lambda):
    # targets: (b, num_transitions, n_agents) 
    # rewards, terminated and mask: (b, num_transitions, 1)

    # Initialise  last  lambda -return  for  not  terminated  episodes
    returns = target_qs.new_zeros(*target_qs.shape)
    returns[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    
    # Backwards  recursive  update  of the "forward  view"
    for t in range(returns.shape[1] - 2, -1,  -1):
        returns[:, t] = td_lambda * gamma * returns[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    
    # Returns lambda-return from t=0 to t=T-1, -> (b, num_transitions - 1, n_agents)
    return returns[:, 0:-1]