import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class QMixerCNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = args.hypernet_embed

        self.num_channels = self.state_shape[0]
        self.width = self.state_shape[1]
        self.height = self.state_shape[2]
        pooling_size = 4 if self.width == 41 else 2
        in_feats = (self.width // pooling_size) * (self.height // pooling_size)
        
        self.hyper_w_1 = nn.Sequential(
            nn.Conv2d(self.num_channels, hypernet_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hypernet_embed, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Flatten(),
            nn.Linear(in_feats, self.embed_dim * self.n_agents)
        )
        
        self.hyper_w_final = nn.Sequential(
            nn.Conv2d(self.num_channels, hypernet_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hypernet_embed, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Flatten(),
            nn.Linear(in_feats, self.embed_dim)
        )
        
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Sequential(
            nn.Conv2d(self.num_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Flatten(),
            nn.Linear(in_feats, self.embed_dim)
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Conv2d(self.num_channels, hypernet_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hypernet_embed, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Flatten(),
            nn.Linear(in_feats, 1)
        )


    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, *self.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(bs, 1, 1)
        return q_tot
