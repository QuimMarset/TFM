import torch as th
from critic_controllers.factorized_controller import FactorizedCriticController



class DDPGController(FactorizedCriticController):

    # Single-agent ddpg with stochastic policies


    def _get_history_shape(self, scheme):
        return scheme["state"]["vshape"]

    
    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0]

    
    def forward(self, ep_batch, t, actions=None):
        history = self._build_history(ep_batch, t)
        actions = self._build_actions(ep_batch, t, actions)
        critic_outs, self.hidden_states = self.critic(history, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_history(self, batch, t):
        return batch["state"][:, t]


    def _build_actions(self, batch, t, actions):
        if actions is None:
            return batch['actions'][:, t]
        return actions