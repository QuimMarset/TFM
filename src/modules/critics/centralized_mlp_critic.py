import torch.nn as nn
import torch as th


class CentralizedMLPCritic(nn.Module):


    def __init__(self, input_shape, action_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape + action_shape * self.n_agents, args.hidden_dim_critic)
        self.fc2 = nn.Linear(args.hidden_dim_critic, args.hidden_dim_critic)
        self.fc3 = nn.Linear(args.hidden_dim_critic, args.n_critic_net_outputs)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim_critic).zero_()


    def forward(self, inputs, hidden_states, actions):
        # inputs -> (b, state_shape)
        # hidden_states -> (b, hidden_dim) but is not used
        # actions -> (b, n_agents, action_shape)

        # (b, n_agents, action_shape) -> (b, n_agents * action_shape)
        actions = actions.view(actions.size(0), -1)

        inputs = th.cat([inputs, actions], dim=-1)

        x = th.relu(self.fc1(inputs))
        x = th.relu(self.fc2(x))
        q = self.fc3(x)

        q = q.view(-1, 1, 1)
        return q, hidden_states