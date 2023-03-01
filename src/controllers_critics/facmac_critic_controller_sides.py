import torch as th
from modules.critics.facmac import FACMACCritic



class FACMACCriticController:


    def __init__(self, scheme, args):
        self.scheme = scheme
        self.args = args
        self.n_agents = args.n_agents
        self.n_shared = self.n_agents - 2

        input_shape = self._get_input_shape(scheme)
        
        if args.obs_agent_id:
            side_input_shape = input_shape - self.n_shared
        else:
            side_input_shape = input_shape

        self.side_1_critic = FACMACCritic(scheme, args, side_input_shape)
        self.side_2_critic = FACMACCritic(scheme, args, side_input_shape)
        self.shared_critic = FACMACCritic(scheme, args, input_shape)

        self.shared_hidden_states = None 
        self.side_1_hidden_states = None 
        self.side_2_hidden_states = None 


    def forward(self, ep_batch, t, actions=None):
        side_1_inputs, shared_inputs, side_2_inputs = self._build_inputs(ep_batch, t, actions)

        side_1_outs, self.side_1_hidden_states = self.side_1_critic(side_1_inputs, actions=None, hidden_state=self.side_1_hidden_states)
        shared_outs, self.shared_hidden_states = self.shared_critic(shared_inputs, actions=None, hidden_state=self.shared_hidden_states)
        side_2_outs, self.side_2_hidden_states = self.side_2_critic(side_2_inputs, actions=None, hidden_state=self.side_2_hidden_states)

        if ep_batch.batch_size > 1:
            side_1_outs = side_1_outs.unsqueeze(dim=1)
            shared_outs = shared_outs.reshape((ep_batch.batch_size, self.n_shared, -1))
            side_2_outs = side_2_outs.unsqueeze(dim=1)
            critic_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=1)
            critic_outs = critic_outs.view(ep_batch.batch_size * self.n_agents, -1)
        else:
            critic_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=0)

        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def init_hidden(self, batch_size):
        self.side_1_critic.init_hidden()
        self.shared_critic.init_hidden()
        self.side_2_critic.init_hidden()


    def parameters(self):
        parameters = []
        for critic in [self.side_1_critic, self.shared_critic, self.side_2_critic]:
            parameters.extend(critic.parameters())
        return parameters
    

    def load_state(self, other_mac):
        self.shared_critic.load_state_dict(other_mac.shared_critic.state_dict())
        self.side_1_critic.load_state_dict(other_mac.side_1_critic.state_dict())
        self.side_2_critic.load_state_dict(other_mac.side_2_critic.state_dict())


    def save_models(self, path):
        th.save(self.shared_critic.state_dict(), f'{path}/shared_critic.th')
        th.save(self.side_1_critic.state_dict(), f'{path}/side_1_critic.th')
        th.save(self.side_2_critic.state_dict(), f'{path}/side_2_critic.th')


    def load_models(self, path):
        self.shared_critic.load_state_dict(th.load(f'{path}/shared_critic.th', map_location=lambda storage, loc: storage))
        self.side_1_critic.load_state_dict(th.load(f'{path}/side_1_critic.th', map_location=lambda storage, loc: storage))
        self.side_2_critic.load_state_dict(th.load(f'{path}/side_2_critic.th', map_location=lambda storage, loc: storage))
    

    def cuda(self, device='cuda'):
        self.side_1_critic.cuda(device=device)
        self.shared_critic.cuda(device=device)
        self.side_2_critic.cuda(device=device)


    def _build_inputs_common(self, batch, t, start_index, end_index, actions, is_shared):
        batch_size = batch.batch_size
        inputs = []
        inputs.append(batch['obs'][:, t, start_index:end_index])

        if actions is None:
            # Used to compute the Q values of the episode saved in the batch
            actions = batch['actions']

        if len(actions.shape) == 3:
                actions = actions.unsqueeze(1)
                inputs.append(actions[:, 0, start_index:end_index])
        else:
            inputs.append(actions[:, t, start_index:end_index])

        if self.args.obs_agent_id and is_shared:
            agent_ids = th.eye(self.n_shared, device=batch.device).unsqueeze(0).expand(batch_size, -1, -1)
            inputs.append(agent_ids)

        if is_shared:
            reshape_dim = [batch_size * self.n_shared, -1]
        else:
            reshape_dim = [batch_size, -1]

        inputs = th.cat([x.reshape(*reshape_dim) for x in inputs], dim=-1)
        return inputs


    def _build_inputs_shared(self, batch, t, actions):
        return self._build_inputs_common(batch, t, 1, -1, actions, is_shared=True)


    def _build_inputs_side_1(self, batch, t, actions):
        return self._build_inputs_common(batch, t, 0, 1, actions, is_shared=False)


    def _build_inputs_side_2(self, batch, t, actions):
        return self._build_inputs_common(batch, t, -1, self.n_agents, actions, is_shared=False)


    def _build_inputs(self, batch, t, actions):
        side_1_inputs = self._build_inputs_side_1(batch, t, actions)
        shared_inputs = self._build_inputs_shared(batch, t, actions)
        side_2_inputs = self._build_inputs_side_2(batch, t, actions)
        return side_1_inputs, shared_inputs, side_2_inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_shared
        return input_shape
