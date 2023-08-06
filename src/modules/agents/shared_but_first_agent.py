import torch.nn as nn
import torch as th



class SharedButFirstAgent(nn.Module):

    # Used in Swimmer
    # Defines 2 agents: 1 to control the first set of joints, 
    # and a second to control the rest extreme pistons, 

    def __init__(self, input_shape, args, agent_class):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._process_input_shape(input_shape)
        self.agents = th.nn.ModuleList([
            agent_class(input_shape, args),
            agent_class(input_shape, args)
        ])


    def _process_input_shape(self, input_shape):
        if self.args.add_agent_id:
            return input_shape - self.n_agents
        return input_shape


    def init_hidden(self):
        # First -> (1, hidden_dim), Shared -> (n_agents - 1, hidden_dim)
        first_hiddens = self.agents[0].init_hidden()
        shared_hiddens = self.agents[1].init_hidden().expand(self.n_agents - 1, -1)
        return th.cat([first_hiddens, shared_hiddens], dim=0)
    

    def _process_inputs(self, side_inputs):
        # We assume the agent IDs are appended at the end
        if self.args.add_agent_id:
            return side_inputs[:, :, :-self.n_agents]
        return side_inputs


    def forward(self, inputs, hidden_states):
        # Both inputs and hidden_state always have 3 dimensions (b, n_agents, -1)
        inputs = self._process_inputs(inputs)
        # Both (b, 1, -1)
        out_side_1, h_side_1 = self.agents[0](inputs[:, 0:1], hidden_states[:, 0:1])
        # Both (b, n_agents - 1, -1)
        out_shared, h_shared = self.agents[1](inputs[:, 1:], hidden_states[:, 1:])

        outs = th.cat([out_side_1, out_shared], dim=1)
        hiddens = th.cat([h_side_1, h_shared], dim=1)
        return outs, hiddens


    def cuda(self, device="cuda:0"):
        for agent in self.agents:
            agent.cuda(device=device)
