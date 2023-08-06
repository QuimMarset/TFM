import torch as th
import torch.distributions as tdist
from controllers.multi_agent.FACMAC.facmac_controller import FACMACAgentController



class ContinuousQController(FACMACAgentController):


    def _get_input_shape(self, scheme):
        return super()._get_input_shape(scheme) + self.args.action_shape


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = self.cem_sampling(ep_batch, t_ep, bs)
        if not test_mode:
            actions = self._add_noise(actions, t_env)
        return actions


    def forward(self, ep_batch, t, actions=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        if actions is not None:
            # Both have shape (b, n_agents, -1)
            agent_inputs = th.cat([agent_inputs, actions], dim=-1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)                
        return agent_outs
    

    def cem_sampling(self, ep_batch, t, bs):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6
        maxits = 2

        ftype = th.FloatTensor if not next(self.agent.parameters()).is_cuda else th.cuda.FloatTensor
        mu = ftype(ep_batch.batch_size, self.n_agents, self.args.action_shape).zero_()
        std = ftype(ep_batch.batch_size, self.n_agents, self.args.action_shape).zero_() + 1.0
        its = 0

        # (b, n_agents, -1)
        history_inputs = self._build_inputs(ep_batch, t)
        # (N, b, n_agents, -1)
        hidden_states = self.hidden_states.reshape(-1, self.n_agents, self.args.hidden_dim).repeat(N, 1, 1, 1)

        while its < maxits:
            dist = tdist.Normal(mu, std)
            # (N, b, n_agents, action_shape)
            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)

            # (N * b, n_agents, -1)
            agent_inputs = history_inputs.unsqueeze(0).expand(N, *history_inputs.shape).contiguous().view(-1, *history_inputs.shape[1:])
            # (N * b, n_agents, action_shape)
            actions_inputs = actions_prime.view(-1, *actions_prime.shape[2:])
            # (N * b, n_agents, input_shape + action_shape)
            agent_inputs = th.cat([agent_inputs, actions_inputs], dim=-1)

            # (N * b, n_agents, 1) and (N * b, n_agents, -1)
            agent_outs, hidden_states_temp = self.agent(agent_inputs, hidden_states)
            # (N, b * n_agents, 1)
            out = agent_outs.view(N, ep_batch.batch_size * self.n_agents, 1)
            
            # (N, b * n_agents, 1)
            topk, topk_idxs = th.topk(out, Ne, dim=0)
            # (N, b * n_agents, action_shape)
            actions = actions.view(N, -1, self.args.action_shape)

            # (b * n_agents, -1)
            mu = th.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.args.action_shape).long()), dim=0)
            std = th.std(actions.gather(0, topk_idxs.repeat(1, 1, self.args.action_shape).long()), dim=0)
            # (b, n_agents, -1)
            mu = mu.view(ep_batch.batch_size, self.n_agents, -1)
            std = std.view(ep_batch.batch_size, self.n_agents, -1)

            if th.any(std < 0):
                std = th.zeros_like(std) + 1

            its += 1

        # (N, b, n_agents, 1)
        topk, topk_idxs = th.topk(out.view(N, ep_batch.batch_size, self.n_agents, 1), 1, dim=0)
        
        # (b, n_agents, -1)
        hidden_states_temp = hidden_states_temp.view(N, ep_batch.batch_size, self.n_agents, -1)
        self.hidden_states = hidden_states_temp.gather(0, topk_idxs.repeat(1, 1, 1, self.args.hidden_dim).long())[0]
        
        # (b, n_agents, action_shape)
        action_prime = actions_prime.gather(0, topk_idxs.repeat(1, 1, 1, self.args.action_shape).long())[0]
        chosen_actions = action_prime.clone().detach()
        return chosen_actions[bs]
    