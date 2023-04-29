import torch.nn as nn
import torch as th



class MLPActorAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(MLPActorAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_net_outputs)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()


    def forward(self, inputs, hidden_state, actions=None):
        # inputs always has 3 dimensions (b, n_agents, -1)
        batch_size = inputs.size(0)
        n_agents = inputs.size(1)
        inputs = inputs.reshape(batch_size * n_agents, -1)

        x = th.relu(self.fc1(inputs))
        x = th.relu(self.fc2(x))
        actions = th.tanh(self.fc3(x))

        actions = actions.view(batch_size, n_agents, -1)
        return actions, hidden_state
