import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
from controllers.maddpg_controller import MADDPGMAC



class MADDPGDistributionMAC(MADDPGMAC):

    def __init__(self, scheme, args):
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        means, log_stds = self.forward(ep_batch, t_ep)
        means = means[bs]
        log_stds = log_stds[bs]
        
        diag_cov_matrix = th.diag_embed(th.exp(log_stds))
        distrib = MultivariateNormal(means, diag_cov_matrix)
        actions = distrib.rsample()
        actions = self._clamp_actions(actions)
        return actions


    def select_actions_with_log_probs(self, ep_batch, t_ep):
        means, log_stds = self.forward(ep_batch, t_ep)

        diag_cov_matrix = th.diag_embed(th.exp(log_stds))
        distrib = MultivariateNormal(means, diag_cov_matrix)
        actions = distrib.rsample()
        actions = self._clamp_actions(actions)
        log_probs = distrib.log_prob(actions).unsqueeze(-1)
        return actions, log_probs


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        means_log_stds, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        means = means_log_stds[:, :, :self.args.action_shape]
        log_stds = means_log_stds[:, :, self.args.action_shape:]

        means = means.view(ep_batch.batch_size, self.n_agents, -1)
        log_stds = log_stds.view(ep_batch.batch_size, self.n_agents, -1)
        return means, log_stds
    