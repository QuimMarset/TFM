from modules.critics import critic_factory
import torch as th



class BaseCriticController:


    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_critics(scheme)


    def _build_critics(self, scheme):
        input_shape = self._get_input_shape(scheme)
        action_shape = self._get_action_shape(scheme)
        kwargs = {
            'input_shape' : input_shape,
            'action_shape' : action_shape,
            'args' : self.args
        }
        self.critic = critic_factory.build(self.args.critic, **kwargs)


    def _get_input_shape(self, scheme):
        raise NotImplementedError()
    

    def _get_action_shape(self, scheme):
        raise NotImplementedError()

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        # (1, h) -> (b, 1, h)
        self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
        # If the MAC shares parameters: (b, 1, h) -> (b, n_agents, h)
        if self.hidden_states.shape[1] != self.n_agents:
            self.hidden_states = self.hidden_states.expand(-1, self.n_agents, -1)


    def forward(self, ep_batch, t):
        raise NotImplementedError()


    def _build_inputs(self, batch, t):
        raise NotImplementedError()
    

    def parameters(self):
        return self.critic.parameters()


    def named_parameters(self):
        return self.critic.named_parameters()


    def load_state(self, other_mac):
        self.critic.load_state_dict(other_mac.critic.state_dict())


    def load_state_from_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict)


    def cuda(self, device="cuda"):
        self.critic.cuda(device=device)


    def save_models(self, path, is_target=False):
        name = 'critic' if not is_target else 'target_critic'
        th.save(self.critic.state_dict(), f'{path}/{name}.th')


    def load_models(self, path, is_target=False):
        name = 'critic' if not is_target else 'target_critic'
        self.critic.load_state_dict(th.load(f'{path}/{name}.th', 
                                           map_location=lambda storage, 
                                           loc: storage))
        