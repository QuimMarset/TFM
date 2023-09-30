import torch as th
import numpy as np
from components.action_selectors import REGISTRY as action_selector_registry
from controllers.base_classes.base_controller import BaseController



class DQNController(BaseController):

    """
        DQN is a single-agent value-based discrete method. Therefore, the action-value network
        needs to output Q-values for all possible actions. We are dealing with environments
        with multiple agents. Therefore, if each agent can take n_actions discrete actions,
        and we have n_agents, the network will handle all the agent actions, outputting 
        n_actions**n_agents Q-values.
        
        The problem is that we need to select action with maximal Q-value both in index and
        binary form. 
        Imagine we have binary actions, 10 agents, and the maximal action is all 1. In that case,
        we need to output both 1023 and 1111111111
    """

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_selector = action_selector_registry[args.action_selector](args)
        self.init_hidden(self.args.batch_size)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # (b, n_discrete_actions ** n_agents)
        agent_outputs = self.forward(ep_batch, t_ep)
        # (b, 1)
        decimal_actions = self.action_selector.select_action(agent_outputs, t_env, test_mode=test_mode)
        # (b, n_agents, 1)
        one_hot_actions = self._decimal_to_binary(decimal_actions.squeeze(-1)).unsqueeze(-1)
        return one_hot_actions[bs]


    def _decimal_to_binary(self, index_actions):
        if self.args.use_cuda:
            index_actions = index_actions.cpu()

        binary_repre = np.vectorize(np.binary_repr)
        binary_actions = binary_repre(index_actions.numpy(), width=self.n_agents).tolist()
        binary_list = lambda x: [int(digit) for digit in x]
        binary_actions = list(map(binary_list, binary_actions))
        return th.tensor(binary_actions, dtype=int, device=self.args.device)

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        # (b, 1, n_discrete_actions ** n_agents), (b, 1, hidden_dim)
        agent_outputs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outputs


    def _build_inputs(self, batch, t):
        # (b, 1, state_shape)
        return batch["state"][:, t].unsqueeze(1)


    def _get_input_shape(self, scheme):
        return scheme["state"]["vshape"]
    