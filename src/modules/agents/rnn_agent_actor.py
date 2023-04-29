import torch.nn as nn
import torch as th



class RNNActorAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNActorAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_net_outputs)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()


    def forward(self, inputs, hidden_state, actions=None):
        # Both inputs and hidden_state always have 3 dimensions (b, n_agents, -1)
        batch_size = inputs.size(0)
        n_agents = inputs.size(1)

        inputs_in = inputs.reshape(batch_size * n_agents, -1)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)

        x = th.relu(self.fc1(inputs_in))
        # Both (b * -1, -1)
        h = self.rnn(x, h_in)
        actions = th.tanh(self.fc2(h))

        h = h.view(batch_size, n_agents, -1)
        actions = actions.view(batch_size, n_agents, -1)
        return actions, h