from critic_controllers.base_controller import BaseCriticController



class DDPGController(BaseCriticController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_shape = args.action_shape
        self.init_hidden(args.batch_size)

    
    def forward(self, ep_batch, t, actions):
        # actions -> (b, 1, n_agents * action_shape)
        # (b, 1, state_shape)
        inputs = self._build_inputs(ep_batch, t)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, 1, 1)


    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
    

    def _build_inputs(self, batch, t):
        # (b, 1, state_shape)
        return batch["state"][:, t].unsqueeze(1)
    
    
    def _get_input_shape(self, scheme):
        return scheme["state"]["vshape"]
    

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0] * self.n_agents
    