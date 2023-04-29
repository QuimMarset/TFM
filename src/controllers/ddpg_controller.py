import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
from controllers.dqn_controller import DQNController



class DDPGController(DQNController):

    # Single-agent DDPG with stochastic policies

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        means, log_stds = self.forward(ep_batch, t_ep)
        batch_size = ep_batch.batch_size

        if test_mode:
            # Only used when evaluating the agent, not during training
            # (b, 1, action_shape * n_agents) -> (b, n_agents, action_shape)
            actions = means.view(batch_size, self.n_agents, -1)
        else:
            diag_cov_matrix = th.diag_embed(th.exp(log_stds))
            distrib = MultivariateNormal(means, diag_cov_matrix)
            # (b, 1, action_shape * n_agents) -> (b, n_agents, action_shape)
            actions = distrib.rsample().view(batch_size, self.n_agents, -1)

        actions = self._clamp_actions(actions)        
        return actions[bs]

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)


    def forward(self, ep_batch, t):
        # (b, 1, -1)
        agent_inputs = self._build_inputs(ep_batch, t).unsqueeze(1)
        means_log_stds, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        # Both (b, 1, action_shape * n_agents)
        means = means_log_stds[:, :, :self.args.action_shape]
        log_stds = means_log_stds[:, :, self.args.action_shape:]

        means = means.view(ep_batch.batch_size, 1, -1)
        log_stds = log_stds.view(ep_batch.batch_size, 1, -1)
        return means, log_stds
    

    def _clamp_actions(self, actions):
        for index in range(self.n_agents):
            action_space = self.args.action_spaces[index]
            for dimension_num in range(action_space.shape[0]):
                min_action = action_space.low[dimension_num].item()
                max_action = action_space.high[dimension_num].item()
                actions[:, index, dimension_num].clamp_(min_action, max_action)
        return actions