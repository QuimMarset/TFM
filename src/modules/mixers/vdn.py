import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self, args):
        super(VDNMixer, self).__init__()
        self.args = args


    def forward(self, agent_qs, batch):
        if agent_qs.shape[-1] == 1:
            agent_qs = agent_qs.squeeze(-1)
        joint_qs = th.sum(agent_qs, dim=-1, keepdim=True)
        return joint_qs