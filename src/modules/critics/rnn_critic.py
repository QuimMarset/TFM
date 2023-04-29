import torch as th
import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent




class RNNCritic(RNNAgent):

    def __init__(self, history_shape, action_shape, args):
        input_shape = history_shape + action_shape
        super().__init__(input_shape, args)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_critic_net_outputs)


    def forward(self, history, hidden_state, actions):
        # The three inputs always have 3 dimensions (b, n_agents, -1)
        inputs = th.cat([history, actions], dim=-1)
        return super().forward(inputs, hidden_state)
