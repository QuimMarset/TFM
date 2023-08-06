import torch as th
from modules.agents.shared_but_sides_agent import SharedButSidesAgent



class SharedButSidesCritic(SharedButSidesAgent):

    # Used in the pistonball environment
    # Defines 3 agents: 2 to control the extreme pistons, 
    # and a 3rd to control the ones in between

    def __init__(self, input_shape, action_shape, args, agent_class):
        th.nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        side_shape = self._compute_side_input_shape(input_shape)
        self.agents = th.nn.ModuleList(
            [agent_class(side_shape, action_shape, args),
             agent_class(input_shape, action_shape, args),
             agent_class(side_shape, action_shape, args)
             ])
    

    def _forward_side(self, agent, side_history, side_hidden_states, actions):
        side_history = self._process_side_inputs(side_history)
        # All three -> (b, 1, -1)
        out_side, h_side = agent(side_history, side_hidden_states, actions)
        return out_side, h_side


    def _forward_shared(self, shared_history, shared_hidden_states, actions):
        # All three -> (b, n_agents - 2, -1)
        out_shared, h_shared = self.agents[1](shared_history, shared_hidden_states, actions)
        return out_shared, h_shared


    def forward(self, history, hidden_states, actions):
        # All three inputs always have 3 dimensions (b, n_agents, -1)
        # Both (b, 1, -1)
        out_side_1, h_side_1 = self._forward_side(self.agents[0], history[:, 0:1], hidden_states[:, 0:1], actions[:, 0:1])
        # Both (b, n_agents - 2, -1)
        out_shared, h_shared = self._forward_shared(history[:, 1:-1], hidden_states[:, 1:-1], actions[:, 1:-1])
        # Both (b, 1, -1)
        out_side_2, h_side_2 = self._forward_side(self.agents[-1], history[:, -1:], hidden_states[:, -1:], actions[:, -1:])

        outs = th.cat([out_side_1, out_shared, out_side_2], dim=1)
        hiddens = th.cat([h_side_1, h_shared, h_side_2], dim=1)
        return outs, hiddens
