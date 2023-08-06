from controllers.multi_agent.QMIX.q_controller import QController



class TransformerController(QController):

        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if test_mode:
            # Ensure Batch Norm layers behave in test mode
            self.agent.eval()

        q_vals = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(q_vals[bs], t_env, test_mode=test_mode)
        return chosen_actions.unsqueeze(-1)


    def forward(self, ep_batch, t, return_hs=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if return_hs:
            return agent_outs, self.hidden_states
        else:
            return agent_outs
        