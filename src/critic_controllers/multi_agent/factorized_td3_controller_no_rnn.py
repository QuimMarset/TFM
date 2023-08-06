from critic_controllers.multi_agent.factorized_controller_no_rnn import FactorizedCriticControllerNoRNN
import torch as th



class FactorizedTD3CriticControllerNoRNN(FactorizedCriticControllerNoRNN):


    def forward(self, ep_batch, t, actions):
        # (b, n_agents, -1)
        inputs = self._build_inputs(ep_batch, t)

        critic_1_outs, critic_2_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        critic_1_outs = critic_1_outs.view(ep_batch.batch_size, self.n_agents, 1)
        critic_2_outs = critic_2_outs.view(ep_batch.batch_size, self.n_agents, 1)

        return critic_1_outs, critic_2_outs


    def forward_first(self, ep_batch, t, actions):
        # (b, n_agents, -1)
        input = self._build_inputs(ep_batch, t)

        critic_1_outs, _, self.hidden_states = self.critic(input, self.hidden_states, actions)
        return critic_1_outs.view(ep_batch.batch_size, self.n_agents, 1)
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        if self.hidden_states is not None:
            # (2, 1, h) -> (2, b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(1).expand(-1, batch_size, -1, -1)
            # If the MAC shares parameters: (2, b, 1, h) -> (2, b, n_agents, h)
            if self.hidden_states.shape[2] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, -1, self.n_agents, -1)
