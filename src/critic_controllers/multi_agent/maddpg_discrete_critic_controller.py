from critic_controllers.base_critic_controller import BaseCriticController



class MADDPGDiscreteCriticController(BaseCriticController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.init_hidden(args.batch_size)
        self.n_discrete_actions = args.n_discrete_actions
        self.state_shape = scheme['state']['vshape']


    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]       
        return input_shape
    

    def _get_action_shape(self, scheme):
        return scheme["actions_onehot"]["vshape"][0]


    def forward(self, ep_batch, actions):
        # (b * (num_transitions + 1), n_agents, n_discrete_actions)
        actions = actions.view(-1, self.n_agents, self.n_discrete_actions)
        inputs = self._build_inputs(ep_batch)
        # (b * (num_transitions + 1), 1, 1), (b * (num_transitions + 1), hidden_dim)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, ep_batch.max_seq_length, 1, 1)
    

    def _build_inputs(self, batch):
        # (b * (num_transitions + 1), state_shape)
        return batch['state'].view(-1, self.state_shape)
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        # (1, h) -> (b, h)
        self.hidden_states = self.hidden_states.expand(batch_size, -1)
    