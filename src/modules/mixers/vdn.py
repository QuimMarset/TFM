import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self, args):
        super(VDNMixer, self).__init__()
        self.args = args


    def forward(self, agent_qs, batch):
        shape = agent_qs.shape
        agent_qs = agent_qs.view(-1, self.args.n_agents, 1)
        joint_qs = th.sum(agent_qs, dim=1, keepdim=True)
        return joint_qs.view(*shape[:-2], 1, 1)