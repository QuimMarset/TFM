import torch.nn as nn
import torch as th



class NonSharedAgent(nn.Module):

    # Builds a set of networks, each controlling an agent

    def __init__(self, input_shape, args, agent_class):
        super(NonSharedAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._process_input_shape(input_shape)
        self.agents = th.nn.ModuleList([agent_class(input_shape, args) for _ in range(self.n_agents)])


    def _process_input_shape(self, input_shape):
        # Remove the agent IDs if added (non-shared agents do not need it)
        if self.args.add_agent_id:
            return input_shape - self.n_agents
        return input_shape


    def init_hidden(self):
        # (n_agents, -1)
        return th.cat([agent.init_hidden() for agent in self.agents])
    

    def _process_inputs(self, inputs):
        # Assume agent IDs at the end
        if self.args.add_agent_id:
            return inputs[:, :, :-self.n_agents]
        return inputs


    def forward(self, inputs, hidden_state):
        # Both inputs and hidden_state always have 3 dimensions (b, n_agents, -1)
        inputs = self._process_inputs(inputs)
        hiddens = []
        outs = []
        for i, agent in enumerate(self.agents):
            # (b, 1, -1) -> By indexing with the : we keep the agent dimension
            # Both (b, 1, -1)
            out, h = agent(inputs[:, i:i+1], hidden_state[:, i:i+1])
            hiddens.append(h)
            outs.append(out)
        return th.cat(outs, dim=1), th.cat(hiddens, dim=1)


    def cuda(self, device="cuda:0"):
        for agent in self.agents:
            agent.cuda(device=device)
