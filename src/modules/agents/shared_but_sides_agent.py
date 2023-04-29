import torch.nn as nn
import torch as th



class SharedButSidesAgent(nn.Module):

    # Used in the pistonball environment
    # Defines 3 agents: 2 to control the extreme pistons, 
    # and a 3rd to control the ones in between

    def __init__(self, input_shape, args, agent_class):
        super(SharedButSidesAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        side_shape = self._compute_side_input_shape(input_shape)
        self.agents = th.nn.ModuleList(
            [agent_class(side_shape, args),
             agent_class(input_shape, args),
             agent_class(side_shape, args)
             ])
        

    def _compute_side_input_shape(self, input_shape):
        if self.args.obs_agent_id:
            return input_shape - self.n_agents
        return input_shape


    def init_hidden(self):
        # Sides -> (1, -1), Shared -> (n_agents - 2, -1)
        side_1_hiddens = self.agents[0].init_hidden()
        shared_hiddens = self.agents[1].init_hidden().expand(self.n_agents - 2, -1)
        side_2_hiddens = self.agents[2].init_hidden()
        return th.cat([side_1_hiddens, shared_hiddens, side_2_hiddens], dim=0)


    def _process_side_inputs(self, side_inputs):
        # We assume the agent IDs are appended at the end
        if self.args.obs_agent_id:
            # (b, 1, -1)
            return side_inputs[:, :, :-self.n_agents]
        return side_inputs
    

    def _forward_side(self, agent, side_inputs, side_hidden_states):
        side_inputs = self._process_side_inputs(side_inputs)
        # Both -> (b, 1, -1)
        out_side, h_side = agent(side_inputs, side_hidden_states)
        return out_side, h_side


    def _forward_shared(self, shared_inputs, shared_hidden_states):
        # Both -> (b, n_agents - 2, -1)
        out_shared, h_shared = self.agents[1](shared_inputs, shared_hidden_states)
        return out_shared, h_shared


    def forward(self, inputs, hidden_states):
        # Both inputs and hidden_state always have 3 dimensions (b, n_agents, -1)
        # Both (b, 1, -1)
        out_side_1, h_side_1 = self._forward_side(self.agents[0], inputs[:, 0:1], hidden_states[:, 0:1])
        # Both (b, n_agents - 2, -1)
        out_shared, h_shared = self._forward_shared(inputs[:, 1:-1], hidden_states[:, 1:-1])
        # Both (b, 1, -1)
        out_side_2, h_side_2 = self._forward_side(self.agents[-1], inputs[:, -1:], hidden_states[:, -1:])

        outs = th.cat([out_side_1, out_shared, out_side_2], dim=1)
        hiddens = th.cat([h_side_1, h_shared, h_side_2], dim=1)
        return outs, hiddens


    def cuda(self, device="cuda:0"):
        for agent in self.agents:
            agent.cuda(device=device)
