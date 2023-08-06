import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CustomMixer(nn.Module):


    def __init__(self, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.mixer_hidden_dim = args.mixing_embed_dim

        self.fc1 = nn.Linear(self.n_agents, self.mixer_hidden_dim)
        self.fc2 = nn.Linear(self.mixer_hidden_dim, self.mixer_hidden_dim)
        self.fc3 = nn.Linear(self.mixer_hidden_dim + self.state_dim, 1)


    def forward(self, agent_qs, states):
        # agent_qs: (b, n_agents, 1) -> (b, n_agents)
        agent_qs = agent_qs.view(-1, self.n_agents)
        # states: (b, 1, state_shape) -> (b, state_shape)
        states = states.view(-1, self.state_dim)

        #inputs = th.cat([agent_qs, states], dim=-1)

        x = th.relu(self.fc1(agent_qs))
        x = th.relu(self.fc2(x))

        x = th.cat([x, states], dim=-1)

        joint_qs = self.fc3(x)
        return joint_qs.view(-1, 1, 1)
