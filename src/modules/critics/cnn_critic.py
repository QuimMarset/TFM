import torch as th
import torch.nn as nn



class CNNCritic(nn.Module):

    """
        This network works with single-agent and multi-agent methods
        In single-agent methods we generate actions with the same width and height
        as the state. In the multi-agent case, we divide it in as many parts as agents
        Hence, we use max pooling and padding to perform this division
    """

    def __init__(self, input_shape, action_shape, args):
        super().__init__()
        self.args = args
        self.agent_masks = th.tensor(self.args.agent_masks, device=args.device)
        self.n_agents = self.args.n_agents
        
        self.num_channels = input_shape[0]
        self.width = input_shape[1]
        self.height = input_shape[2]

        self.action_width = action_shape[0]
        self.action_height = action_shape[1]

        self.conv1 = nn.Conv2d(self.num_channels + 1, args.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(args.hidden_dim, 1, kernel_size=3, padding=1)
        
        pool_size = 4 if self.width == 41 else 2
        self.max_pool_1 = nn.MaxPool2d(pool_size)
        
        in_feats = (self.width // pool_size) * (self.height // pool_size)
        self.fc1 = nn.Linear(in_feats, 1)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()


    """
        We assume the state input is as follow in the adaptive optics environment
        4 agents:
            A1 | A2
            A4 | A3
        2 agents:
            A1 | A2
        1 agent:
            A1
    """
    def _forward_agent(self, inputs_agent, actions_agent, agent_index):
        # 1, 2 or 4 agents
        angle_multiplier = 2 if self.n_agents == 2 else 1

        # We rotate the state counterclockwise
        if self.n_agents > 1:
            inputs_agent = th.rot90(inputs_agent, agent_index * angle_multiplier, dims=(2, 3))
            actions_agent = th.rot90(actions_agent, agent_index * angle_multiplier, dims=(1, 2))

        if self.n_agents > 1:
            actions = th.zeros((actions_agent.size(0), self.width, self.height), device=self.args.device)
            actions[:, :self.action_width, :self.action_height] = actions_agent
        else:
            actions = actions_agent
            
        x = th.cat([inputs_agent, actions.unsqueeze(1)], dim=1)

        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        x = th.relu(self.conv3(x))
        x = self.max_pool_1(x.squeeze(1))
        x = th.flatten(x, start_dim=1)
        qs = self.fc1(x)
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
