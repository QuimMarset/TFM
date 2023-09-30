from critic_controllers.single_agent.ddpg_critic_controller import DDPGCriticController



class TD3CriticController(DDPGCriticController):


    def forward(self, ep_batch, t, actions):
        # actions -> (b, 1, n_agents * action_shape)
        # (b, 1, state_shape)
        input = self._build_inputs(ep_batch, t)
        # Both (b, 1, 1)        
        critic_1_outs, critic_2_outs, self.hidden_states = self.critic(input, self.hidden_states, actions)
        return critic_1_outs, critic_2_outs
    

    def forward_first(self, ep_batch, t, actions):
        # actions -> (b, 1, n_agents * action_shape)
        # (b, 1, state_shape)
        input = self._build_inputs(ep_batch, t)
        # (b, 1, 1), (b, 1, hidden_dim)
        critic_1_outs, self.hidden_states = self.critic.forward_first(input, self.hidden_states, actions)
        return critic_1_outs
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        if self.hidden_states is not None:
            # (2, 1, h) -> (2, b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(1).expand(-1, batch_size, -1, -1)
            # If the MAC shares parameters: (2, b, 1, h) -> (2, b, n_agents, h)
            if self.hidden_states.shape[2] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, -1, self.n_agents, -1)