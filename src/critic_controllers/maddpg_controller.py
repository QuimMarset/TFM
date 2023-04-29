from modules.critics import critic_factory
import torch as th



class MADDPGCriticController:

    # Used with continuous actions

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_critics(scheme)
        # It is not used, but needed to run the forward
        self.init_hidden(args.batch_size)


    def _build_critics(self, scheme):
        input_shape = self._get_input_shape(scheme)
        action_shape = self._get_action_shape(scheme)
        kwargs = {
            # In MADDPG we do not pass the history of obs-action pairs as we have a centralized
            # and monolithic critic, but we only pass the state
            'history_shape' : input_shape,
            'action_shape' : action_shape,
            'args' : self.args
        }
        self.critic = critic_factory.build(self.args.critic, **kwargs)


    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.critic_obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        if self.args.critic_obs_agent_id:
            # We have as many critics as agents, but we can share parameters
            input_shape += self.n_agents
        return input_shape
    

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0] * self.n_agents

    
    def init_hidden(self, batch_size):
        # Critic is not recurrent, hidden_states always ignored
        self.hidden_states = self.critic.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
            # If the MAC does not share parameters: (b, 1, h) -> (b, n_agents, h)
            if self.hidden_states.shape[1] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, self.n_agents, -1)


    def forward(self, ep_batch, actions=None):
        inputs = self._build_inputs(ep_batch)
        actions = self._build_actions(ep_batch, actions)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, ep_batch.max_seq_length - 1, self.n_agents, -1)


    def _build_inputs(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length - 1

        inputs = []
        inputs.append(batch["state"][:, :max_t].unsqueeze(2).expand(-1, -1, self.n_agents, -1))

        if self.args.critic_obs_individual_obs:
            inputs.append(batch["obs"][:, :max_t])

        if self.args.critic_obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # All inputs have shape (b, t, n_agents, -1)
        inputs = th.cat(inputs, dim=-1)

        return inputs.reshape(bs * max_t, self.n_agents, -1)


    def _build_actions(self, batch, actions):
        batch_size = batch.batch_size
        max_t = batch.max_seq_length - 1
        if actions is None:
            actions = batch['actions']
            # (b, t, n_agents, -1) -> (b, t, 1, n_agents * -1) -> (b, t, n_agents, n_agents * -1)
            actions = actions.view(batch_size, -1, 1, self.n_agents * self.args.action_shape).expand(-1, -1, self.n_agents, -1)
        return actions.reshape(batch_size * max_t, self.n_agents, -1)
    

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
        