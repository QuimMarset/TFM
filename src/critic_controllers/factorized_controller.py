from modules.critics import critic_factory
import torch as th



class FactorizedCriticController:

    # Used with actor-critic methods having continuous actions and factorizing
    # the joint action-value function

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_critics(scheme)


    def _build_critics(self, scheme):
        history_shape = self._get_history_shape(scheme)
        action_shape = self._get_action_shape(scheme)
        kwargs = {
            'history_shape' : history_shape,
            'action_shape' : action_shape,
            'args' : self.args
        }
        self.critic = critic_factory.build(self.args.critic, **kwargs)


    def _get_history_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.critic_obs_last_action:
            # When adding the last action, we assume we pass the history
            input_shape += self._get_action_shape(scheme)
        if self.args.critic_obs_agent_id:
            # When adding the agent ID, we assume we share weights
            input_shape += self.n_agents
        return input_shape
    

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0]

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
            # If the MAC does not share parameters: (b, 1, h) -> (b, n_agents, h)
            if self.hidden_states.shape[1] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, self.n_agents, -1)


    def forward(self, ep_batch, t, actions=None):
        history = self._build_history(ep_batch, t)
        actions = self._build_actions(ep_batch, t, actions)
        critic_outs, self.hidden_states = self.critic(history, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_history(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.critic_obs_last_action:
            # When adding the last action, we assume we pass the history
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])

        if self.args.critic_obs_agent_id:
            # When adding the agent ID, we assume we share weights
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _build_actions(self, batch, t, actions):
        if actions is None:
            return batch['actions'][:, t]
        return actions
    

    def parameters(self):
        return self.critic.parameters()


    def named_parameters(self):
        return self.critic.named_parameters()


    def load_state(self, other_mac):
        self.critic.load_state_dict(other_mac.agent.state_dict())


    def load_state_from_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict)


    def cuda(self, device="cuda"):
        self.critic.cuda(device=device)


    def save_models(self, path):
        th.save(self.critic.state_dict(), f'{path}/agent.th')


    def load_models(self, path):
        self.critic.load_state_dict(th.load(f'{path}/agent.th', 
                                           map_location=lambda storage, 
                                           loc: storage))
        