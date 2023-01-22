from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
import math


class SingleAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.args.n_outputs = self.n_actions ** self.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agent(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        #avail_actions = ep_batch["avail_actions"][:, t_ep]
        avail_actions = th.tensor([1]*self.args.n_outputs).unsqueeze(0).expand(ep_batch.batch_size, -1)
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        processed_actions = self.process_actions(chosen_actions)
        return processed_actions


    def process_actions(self, chosen_actions):
        binary_repre = np.vectorize(np.binary_repr)
        binary_actions = binary_repre(chosen_actions.numpy(), width=self.n_agents).tolist()
        binary_list = lambda x: [int(digit) for digit in x]
        binary_actions = list(map(binary_list, binary_actions))
        return th.tensor(binary_actions, dtype=int)

    
    def init_hidden(self, batch_size):
        pass


    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        outputs = self.agent(inputs)
        return outputs.view(ep_batch.batch_size, -1)


    def parameters(self):
        return self.agent.parameters()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())


    def cuda(self):
        self.agent.cuda()


    def save_models(self, path):
        th.save(self.agent.state_dict(), f'{path}/agent.th')


    def load_models(self, path):
        self.agent.load_state_dict(th.load(f'{path}/agent.th', map_location=lambda storage, loc: storage))


    def _build_agent(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)


    def _build_inputs(self, batch, t):
        return batch["state"][:, t]


    def _get_input_shape(self, scheme):
        return scheme["state"]["vshape"]