from critic_controllers.multi_agent.maddpg_discrete_critic_controller import MADDPGDiscreteCriticController



class MADDPGCriticController(MADDPGDiscreteCriticController):

    # Used with continuous actions

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_shape = args.action_shape


    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"]


    def forward(self, ep_batch, actions):
        # (b * (num_transitions + 1), n_agents, action_shape)
        actions = actions.view(-1, self.n_agents, self.action_shape)
        inputs = self._build_inputs(ep_batch)
        # (b * (num_transitions + 1), 1, 1), (b * (num_transitions + 1), hidden_dim)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, ep_batch.max_seq_length, 1, 1)
