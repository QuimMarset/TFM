import torch.nn as nn
import torch as th
from modules.critics.non_shared_critic import NonSharedCritic



class NonSharedDoubleCritic(nn.Module):

    """ Builds two identic critics to use in TD3 (single-agent or multi-agent)
        In single-agent:
            inputs: (b, 1, state_shape)
            actions: (b, 1, n_agents * action_shape)
            outputs: (b, 1, 1)
        In multi-agent:
            inputs: (b, n_agents, input_shape) -> or obs_shape, plus action ID if needed
            actions: (b, n_agents, action_shape)
            outputs: (b, n_agents, 1)
        In both cases:
            hidden_state: (2, b, n_agents, hidden_dim)
    """

    def __init__(self, input_shape, action_shape, args, agent_class):
        super().__init__()
        self.args = args
        self.critics = th.nn.ModuleList([
            NonSharedCritic(input_shape, action_shape, args, agent_class), 
            NonSharedCritic(input_shape, action_shape, args, agent_class)
        ])


    def init_hidden(self):
        # (2, n_agents, hidden_dim)
        return th.cat([critic.init_hidden().unsqueeze(0) for critic in self.critics])
    

    def forward(self, inputs, hidden_state, actions):
        q_values_1, hidden_state_1 = self.critics[0](inputs, hidden_state[0], actions)
        q_values_2, hidden_state_2 = self.critics[1](inputs, hidden_state[1], actions)

        hidden_states = th.cat([hidden_state_1.unsqueeze(0), hidden_state_2.unsqueeze(0)], dim=0)

        # (b, n_agents, 1), (b, n_agents, 1), (2, b, n_agents, hidden_dim)
        return q_values_1, q_values_2, hidden_states
    

    def forward_first(self, inputs, hidden_state, actions):
        q_values_1, _ = self.critics[0](inputs, hidden_state[0], actions)
        return q_values_1, hidden_state



    def cuda(self, device="cuda:0"):
        for critic in self.critics:
            critic.cuda(device=device)
