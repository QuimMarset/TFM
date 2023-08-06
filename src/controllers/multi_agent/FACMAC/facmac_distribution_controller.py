import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
from controllers.multi_agent.FACMAC.facmac_controller import FACMACAgentController
from controllers.multi_agent.FACMAC.facmac_controller_no_rnn import FACMACAgentControllerNoRNN
from controllers.base_controller import BaseController
from components.action_noising import ActionClamper



class FACMACDistributionAgentController(FACMACAgentController):


    def __init__(self, scheme, args):
        BaseController.__init__(self, scheme, args)
        self.action_clamper = ActionClamper(args.action_spaces, args.device)
        self.init_hidden(args.batch_size)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        means, log_stds = self.forward(ep_batch, t_ep)
        if test_mode:
            # Only used when evaluating the agent, not during training
            actions = means
        else:
            diag_cov_matrix = th.diag_embed(th.exp(log_stds))
            distrib = MultivariateNormal(means, diag_cov_matrix)
            actions = distrib.rsample()
        actions = self.action_clamper.clamp_actions(actions)        
        return actions[bs]
    

    def select_actions_with_log_probs(self, ep_batch, t_ep):
        means, log_stds = self.forward(ep_batch, t_ep)
        
        diag_cov_matrix = th.diag_embed(th.exp(log_stds))
        distrib = MultivariateNormal(means, diag_cov_matrix)
        
        actions = distrib.rsample()
        actions = self.action_clamper.clamp_actions(actions)
        
        log_probs = distrib.log_prob(actions).unsqueeze(-1)
        
        return actions, log_probs
    

    def select_actions_with_entropy(self, ep_batch, t_ep):
        means, log_stds = self.forward(ep_batch, t_ep)
        
        diag_cov_matrix = th.diag_embed(th.exp(log_stds))
        distrib = MultivariateNormal(means, diag_cov_matrix)
        
        actions = distrib.rsample()
        actions = self.action_clamper.clamp_actions(actions)

        entropy = distrib.entropy().mean(dim=-1, keepdim=True)

        return actions, entropy


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        means_log_stds, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        means = means_log_stds[:, :, :self.args.action_shape]
        log_stds = means_log_stds[:, :, self.args.action_shape:]

        means = means.view(ep_batch.batch_size, self.n_agents, -1)
        log_stds = log_stds.view(ep_batch.batch_size, self.n_agents, -1)
        return means, log_stds
    

    def forward_train(self, ep_batch, t):
        means, log_stds = self.forward(ep_batch, t)
        diag_cov_matrix = th.diag_embed(th.exp(log_stds))
        distrib = MultivariateNormal(means, diag_cov_matrix)
        actions = distrib.rsample()
        actions = self.action_clamper.clamp_actions(actions)
        return actions