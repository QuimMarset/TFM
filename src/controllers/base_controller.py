from modules.agents import agent_factory
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th



class BaseController:

    def __init__(self, scheme, args):
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        raise NotImplementedError()


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs


    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
            # If the MAC shares parameters: (b, 1, h) -> (b, n_agents, h)
            if self.hidden_states.shape[1] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, self.n_agents, -1)
        

    def parameters(self):
        return self.agent.parameters()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())


    def load_state_from_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)


    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)


    def _build_agents(self, input_shape):
        kwargs = {
            'input_shape' : input_shape,
            'args' : self.args
        }
        self.agent = agent_factory.build(self.args.agent, **kwargs)


    def _build_inputs(self, batch, t):
        raise NotImplementedError()


    def _get_input_shape(self, scheme):
        raise NotImplementedError()


    def save_models(self, path, is_target=False):
        name = 'agent' if not is_target else 'target_agent'
        th.save(self.agent.state_dict(), f'{path}/{name}.th')


    def load_models(self, path, is_target=False):
        name = 'agent' if not is_target else 'target_agent'
        self.agent.load_state_dict(th.load(f'{path}/{name}.th', 
                                           map_location=lambda storage, 
                                           loc: storage))
        