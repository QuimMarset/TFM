import torch as th
import torch.nn as nn



class CNNActorAgent(nn.Module):

    """
        This network works with single-agent and multi-agent methods
        In single-agent methods we generate actions with the same width and height
        as the state. In the multi-agent case, we divide it in as many parts as agents
        Hence, we use max pooling and padding to perform this division
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        
        self.agent_masks = th.tensor(self.args.agent_masks, device=args.device)
        self.n_agents = self.args.n_agents
        
        num_channels = input_shape[0]
        self.conv1 = nn.Conv2d(num_channels, args.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=3, padding=1)
        
        if self.args.n_agents == 2:
            self.max_pool_1 = nn.MaxPool2d((1, 2))
        else: # n_agents = 4
            self.max_pool_1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(args.hidden_dim, 1, kernel_size=3, padding=1)


    def init_hidden(self):
        return nn.Linear(1, 1).weight.new(1, self.args.hidden_dim).zero_()


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
    def _forward_agent(self, inputs_agent, agent_index):
        # 1, 2 or 4 agents
        angle_multiplier = 2 if self.n_agents == 2 else 1

        # We rotate the state counterclockwise
        inputs_agent = th.rot90(inputs_agent, agent_index * angle_multiplier, dims=(2, 3))
        x = th.relu(self.conv1(inputs_agent))
        x = th.relu(self.conv2(x))
        
        if self.n_agents > 1 and not self.args.env_args['partial_observability']:
            # (11, 11) -> (12, 12) for example
            if self.args.n_agents == 2:
                pad = (1, 0, 0, 0)
            else:
                pad = (1, 0, 1, 0)
            x = nn.functional.pad(x, pad=pad, mode='replicate')
            
            # (12, 12) -> (6, 6)
            x = self.max_pool_1(x)

        # actions (b, 1, width, height) or (b, 1, action_width, action_height)
        agent_actions = th.tanh(self.conv3(x))
        if self.n_agents > 1:
            agent_actions = th.rot90(agent_actions, -agent_index * angle_multiplier, dims=(2, 3))
        agent_actions = agent_actions * self.agent_masks[agent_index]
        return agent_actions


    def forward(self, inputs, hidden_state):
        # inputs has 5 dimensions (b, n_agents, num_channels, width, height)
        # hidden_state is not used
        actions = []

        for i in range(self.n_agents):
            # (b, 1, action_width, action_height)
            actions_agent = self._forward_agent(inputs[:, i], i)
            actions.append(actions_agent)

        actions = th.cat(actions, dim=1)
        return actions, hidden_state
