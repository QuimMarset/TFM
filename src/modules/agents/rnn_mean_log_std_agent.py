import torch.nn as nn
import torch as th



class RNNMeanLogStdAgent(nn.Module):

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.mean = nn.Linear(args.hidden_dim, args.n_net_outputs)
        self.log_std = nn.Linear(args.hidden_dim, args.n_net_outputs)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    

    def forward(self, inputs, hidden_states):
        # inputs always has 3 dimensions (b, n_agents, -1)
        batch_size = inputs.size(0)
        n_agents = inputs.size(1)
        inputs = inputs.reshape(batch_size * n_agents, -1)
        hidden_states = hidden_states.reshape(-1, self.args.hidden_dim)

        x = th.relu(self.fc1(inputs))
        h = self.rnn(x)
        means = self.mean(h)
        log_stds = self.log_std(h)

        means = means.view(batch_size, n_agents, -1)
        log_stds = log_stds.view(batch_size, n_agents, -1)
        # Group it together to ensure always having 2 outputs
        means_log_stds = th.cat([means, log_stds], dim=-1)
        h = h.view(batch_size, n_agents, -1)
        return means_log_stds, h
