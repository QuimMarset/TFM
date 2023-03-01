import torch as th
from modules.critics.facmac import FACMACCritic



class FACMACCriticNonSharedController:


    def __init__(self, scheme, args):
        self.scheme = scheme
        self.args = args
        self.n_agents = args.n_agents
        self.n_shared = self.n_agents - 2

        input_shape = self._get_input_shape(scheme)
        self.critics = [FACMACCritic(scheme, args, input_shape) for _ in range(args.n_agents)]


    def forward(self, ep_batch, t, actions=None):
        critic_inputs = self._build_inputs(ep_batch, t, actions)
        critic_outs = []

        for i, critic in enumerate(self.critics):
            critic_out, _ = critic(critic_inputs[:, i], actions=None)
            critic_outs.append(critic_out)

        critic_outs = th.stack(critic_outs, dim=0)
        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def init_hidden(self, batch_size):
        for critic in self.critics:
            critic.init_hidden()


    def parameters(self):
        parameters = []
        for critic in self.critics:
            parameters.extend(critic.parameters())
        return parameters


    def load_state(self, other_mac):
        for critic, other_critic in zip(self.critics, other_mac.critics):
            critic.load_state_dict(other_critic.state_dict())


    def save_models(self, path):
        for i, critic in enumerate(self.critics):
            th.save(critic.state_dict(), f'{path}/critic_{i}.th')


    def load_models(self, path):
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(th.load(f'{path}/critic_{i}.th', map_location=lambda storage, loc: storage))


    def cuda(self, device='cuda'):
        for critic in self.critics:
            critic.cuda(device=device)


    def _build_inputs(self, batch, t, actions):
        bs = batch.batch_size
        inputs = []

        inputs.append(batch['obs'][:, t])

        if actions is None:
            # Used to compute the Q values of the episode saved in the batch
            actions = batch['actions']

        if len(actions.shape) == 3:
                actions = actions.unsqueeze(1)
                inputs.append(actions[:, 0])
        else:
            inputs.append(actions[:, t])

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        return input_shape
