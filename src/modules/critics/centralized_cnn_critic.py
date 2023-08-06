import torch.nn as nn
import torch as th


class CentralizedCNNCritic(nn.Module):


    def __init__(self, input_shape, action_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        
        num_channels = input_shape[0]
        self.width = input_shape[1]
        self.height = input_shape[2]

        self.conv1 = nn.Conv2d(num_channels + 1, args.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(args.hidden_dim, 1, kernel_size=3, padding=1)
        
        pool_size = 4 if self.width == 41 else 2
        self.max_pool_1 = nn.MaxPool2d(pool_size)

        in_feats = (self.width // pool_size) * (self.height // pool_size)
        self.fc1 = nn.Linear(in_feats, args.n_critic_net_outputs)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()


    def forward(self, inputs, hidden_states, actions):
        # inputs -> (b, num_channels, width, height)
        # hidden_states -> (b, hidden_dim) but not used
        # actions -> (b, n_agents, action_width, action_height)

        # (b, n_agents, action_width, action_height) -> (b, 1, width, height)
        actions = self._merge_agent_actions(actions)

        x = th.cat([inputs, actions], dim=1)

        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        x = th.relu(self.conv3(x))
        x = self.max_pool_1(x)
        x = th.flatten(x, start_dim=1)
        qs = self.fc1(x)
        
        return qs.view(-1, 1, 1), hidden_states
    

    def _get_joint_action_indices(self, agent_index):
        half_side = self.width // 2

        if self.n_agents == 2:
            start_row = 0
            end_row = self.width
            start_col = 0 if agent_index == 0 else half_side
            end_col = half_side + 1 if agent_index == 0 else self.width

        else: # n_agents == 4
            start_row = 0 if agent_index in [0, 1] else half_side
            end_row = start_row + half_side + 1
            start_col = 0 if agent_index in [0, 3] else half_side
            end_col = start_col + half_side + 1
        
        return start_row, end_row, start_col, end_col
     

    def _merge_agent_actions(self, actions):
        joint_actions = th.zeros((actions.size(0), self.width, self.height), device=actions.device)

        for i in range(self.n_agents):
            start_row, end_row, start_col, end_col = self._get_joint_action_indices(i)
            joint_actions[:, start_row:end_row, start_col:end_col] += actions[:, i]

        return joint_actions.unsqueeze(1)