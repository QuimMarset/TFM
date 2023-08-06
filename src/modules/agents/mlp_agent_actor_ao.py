import torch.nn as nn
import torch as th
import numpy as np



class MLPActorAgentAdaptiveOptics(nn.Module):

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.agent_masks = th.tensor(self.args.agent_masks, device=args.device, requires_grad=False)
        
        input_shape = int(np.prod(input_shape))
        action_shape = int(np.prod(args.n_net_outputs))
        
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, action_shape)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    

    """
        We assume the state input is as follow in the adaptive optics environment
        4 agents:       2 agents:       1 agent:   
            A1 | A2         A1 | A2         A1
            A4 | A3
    """
    def _forward_agent(self, inputs_agent, agent_index):
        # 1, 2 or 4 agents
        angle_multiplier = 2 if self.n_agents == 2 else 1

        # We rotate the state counterclockwise
        inputs_agent = th.rot90(inputs_agent, agent_index * angle_multiplier, dims=(2, 3))
        inputs_agent = th.flatten(inputs_agent, start_dim=1)

        x = th.relu(self.fc1(inputs_agent))
        x = th.relu(self.fc2(x))
        # actions (b, width * height) or (b, action_width * action_height)
        agent_actions = th.tanh(self.fc3(x))

        mask_shape = self.agent_masks[agent_index].shape
        agent_actions = agent_actions.reshape(agent_actions.size(0), *mask_shape)
        agent_actions = th.rot90(agent_actions, -agent_index * angle_multiplier, dims=(1, 2))
        agent_actions = agent_actions * self.agent_masks[agent_index]
        return agent_actions


    def forward(self, inputs, hidden_state):
        # inputs has 5 dimensions (b, n_agents, num_channels, width, height)
        actions = []

        for i in range(self.n_agents):
            # (b, action_width, action_height)
            actions_agent = self._forward_agent(inputs[:, i], i)
            actions.append(actions_agent.unsqueeze(1))

        actions = th.cat(actions, dim=1)
        return actions, hidden_state
