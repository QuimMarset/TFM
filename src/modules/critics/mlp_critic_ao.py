import torch as th
import torch.nn as nn
import numpy as np



class MLPCriticAdaptiveOptics(nn.Module):

    def __init__(self, input_shape, action_shape, args):
        super().__init__()
        self.args = args
        self.agent_masks = th.tensor(self.args.agent_masks, device=args.device, requires_grad=False)
        self.n_agents = self.args.n_agents
        self.action_shape = action_shape

        input_shape = int(np.prod(input_shape))
        action_shape = int(np.prod(action_shape))
        
        self.fc1 = nn.Linear(input_shape + action_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_critic_net_outputs)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()


    """
        We assume the state input is as follow in the adaptive optics environment
        4 agents:       2 agents:       1 agent:   
            A1 | A2         A1 | A2         A1
            A4 | A3
    """
    def _forward_agent(self, inputs_agent, actions_agent, agent_index):
        # 1, 2 or 4 agents
        angle_multiplier = 2 if self.n_agents == 2 else 1

        # We rotate the state counterclockwise
        inputs_agent = th.rot90(inputs_agent, agent_index * angle_multiplier, dims=(2, 3))
        inputs_agent = th.flatten(inputs_agent, start_dim=1)

        actions_agent = th.rot90(actions_agent, agent_index * angle_multiplier, dims=(1, 2))
        actions_agent = th.flatten(actions_agent, start_dim=1)

        x = th.cat([inputs_agent, actions_agent], dim=-1)

        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        qs = self.fc3(x)
        return qs


    def forward(self, inputs, hidden_state, actions):
        # inputs has 5 dimensions (b, n_agents, num_channels, width, height)
        # hidden_state is not used
        # actions has 4 dimensions (b, n_agents, action_width, action_height)
        qs = []

        # Apply masks again to ensure we remove wrong actuators
        # It can happen in methods that add noise to the target actions
        actions = actions * self.agent_masks

        for i in range(self.n_agents):
            # (b, 1)
            qs_agent = self._forward_agent(inputs[:, i], actions[:, i], i)
            qs.append(qs_agent.unsqueeze(1))

        qs = th.cat(qs, dim=1)
        return qs.view(-1, self.n_agents, 1), hidden_state
