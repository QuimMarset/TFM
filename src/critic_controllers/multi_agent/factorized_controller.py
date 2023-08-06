import torch as th
from critic_controllers.base_controller import BaseCriticController



class FactorizedCriticController(BaseCriticController):

    # Used with actor-critic methods having continuous actions and factorizing
    # the joint action-value function

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.critic_add_last_action:
            # When adding the last action, we assume we pass the history
            input_shape += self._get_action_shape(scheme)
        if self.args.critic_add_agent_id:
            # When adding the agent ID, we assume we share weights
            input_shape += self.n_agents
        return input_shape
    

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0]


    def forward(self, ep_batch, t, actions):
        history = self._build_inputs(ep_batch, t)
        critic_outs, self.hidden_states = self.critic(history, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.critic_add_last_action:
            # When adding the last action, we assume we pass the history
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])

        if self.args.critic_add_agent_id:
            # When adding the agent ID, we assume we share weights
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
