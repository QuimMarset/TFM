from controllers.multi_agent.QMIX.q_controller import QController



class TransformerController(QController):

        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if test_mode:
            # Ensure Batch Norm layers behave in test mode
            self.agent.eval()

        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(agent_outputs, t_env,
                                                            test_mode=test_mode)
        # (b, n_agents, 1)
        chosen_actions = chosen_actions.unsqueeze(-1)
        return chosen_actions[bs]


    def forward(self, ep_batch, t, return_hs=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # (b, n_agents, n_discrete_actions), (b, n_agents, hidden_dim)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if return_hs:
            return agent_outs, self.hidden_states
        else:
            return agent_outs
        