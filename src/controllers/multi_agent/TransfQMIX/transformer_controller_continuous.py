from controllers.base_classes.base_multi_agent_continuous_controller import BaseMultiAgentContinuousController



class TransformerContinuousController(BaseMultiAgentContinuousController):


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if test_mode:
            # Ensure Batch Norm layers behave in test mode
            self.agent.eval()

        actions = self.forward(ep_batch, t_ep)
        if not test_mode:
            actions = self.action_noising.add_noise(actions, t_env)
            actions = self.action_clamper.clamp_actions(actions)
        return actions[bs]


    def forward(self, ep_batch, t, return_hs=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if return_hs:
            return agent_outs, self.hidden_states
        else:
            return agent_outs
    