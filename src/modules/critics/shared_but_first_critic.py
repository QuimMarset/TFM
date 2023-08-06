import torch as th
from modules.agents.shared_but_first_agent import SharedButFirstAgent



class SharedButFirstCritic(SharedButFirstAgent):

    def __init__(self, input_shape, action_shape, args, agent_class):
        th.nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._process_input_shape(input_shape)
        self.agents = th.nn.ModuleList([
            agent_class(input_shape, action_shape, args),
            agent_class(input_shape, action_shape, args)
        ])
    

    def forward(self, history, hidden_states, actions):
        # All three inputs always have 3 dimensions (b, n_agents, -1)
        history = self._process_inputs(history)

        # Both (b, 1, -1)
        out_side_1, h_side_1 = self.agents[0](history[:, 0:1], hidden_states[:, 0:1], actions[:, 0:1])
        # Both (b, n_agents - 1, -1)
        out_shared, h_shared = self.agents[1](history[:, 1:], hidden_states[:, 1:], actions[:, 1:])

        outs = th.cat([out_side_1, out_shared], dim=1)
        hiddens = th.cat([h_side_1, h_shared], dim=1)
        return outs, hiddens
