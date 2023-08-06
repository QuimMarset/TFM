import torch as th
import numpy as np
from controllers.multi_agent.QMIX.q_controller import QController



class DQNController(QController):


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # (b, n_discrete_actions ** n_agents)
        agent_outputs = self.forward(ep_batch, t_ep)
        # (b, )
        index_actions = self.action_selector.select_action(agent_outputs[bs], 
                                                            t_env, test_mode=test_mode)
        # (b, n_agents, 1)
        one_hot_actions = self.process_actions(index_actions).unsqueeze(-1)
        return one_hot_actions


    def process_actions(self, index_actions):
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
        agent_inputs = self._build_inputs(ep_batch, t).unsqueeze(1)
        agent_outputs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outputs.view(ep_batch.batch_size, -1)


    def _build_inputs(self, batch, t):
        return batch["state"][:, t]


    def _get_input_shape(self, scheme):
        return scheme["state"]["vshape"]
    