import torch as th
from modules.agents.non_shared_agent import NonSharedAgent



class NonSharedCritic(NonSharedAgent):

    # In this sub-class, agent refers to the critic

    def __init__(self, input_shape, action_shape, args, agent_class):
        th.nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._process_input_shape(input_shape)
        self.agents = th.nn.ModuleList([agent_class(input_shape, action_shape, args) for _ in range(self.n_agents)])
    

    def forward(self, history, hidden_state, actions):
        # The three inputs always have 3 dimensions (b, n_agents, -1)
        history = self._process_inputs(history)
        hiddens = []
        outs = []
        for i, agent in enumerate(self.agents):
            # (b, 1, -1) -> By indexing with the : we keep the agent dimension
            # Both (b, 1, -1)
            out, h = agent(history[:, i:i+1], hidden_state[:, i:i+1], actions[:, i:i+1])
            hiddens.append(h)
            outs.append(out)
        return th.cat(outs, dim=1), th.cat(hiddens, dim=1)
