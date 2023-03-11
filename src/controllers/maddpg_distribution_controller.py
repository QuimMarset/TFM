from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
from torch.distributions.normal import Normal
import numpy as np


class MADDPGDistributionMAC:

    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self._build_agents(scheme)
        self.action_selector = None


    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        means, log_stds = self.forward(ep_batch, t_ep)
        distrib = Normal(means, th.exp(log_stds))
        actions = th.tanh(distrib.rsample())
        return actions


    def select_actions_with_log_probs(self, ep_batch, t_ep):
        means, log_stds = self.forward(ep_batch, t_ep)
        distrib = Normal(means, th.exp(log_stds))
        actions = th.tanh(distrib.rsample())
        log_probs = distrib.log_prob(actions)
        return actions, log_probs


    def forward(self, ep_batch, t=None, select_actions=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        means, log_stds = self.agent(agent_inputs)

        if t is None:
            num_steps = ep_batch['state'].shape[1]
            means = means.view(ep_batch.batch_size, num_steps, self.n_agents, -1)
            log_stds = log_stds.view(ep_batch.batch_size, num_steps, self.n_agents, -1)
        else:
            means = means.view(ep_batch.batch_size, self.n_agents, -1)
            log_stds = log_stds.view(ep_batch.batch_size, self.n_agents, -1)

        return means, log_stds


    def init_hidden(self, batch_size):
        pass


    def parameters(self):
        return self.agent.parameters()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())


    def cuda(self, device='cuda'):
        self.agent.cuda(device=device)


    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))


    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))


    def _build_agents(self, scheme):
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        if t is None:
            inputs.append(batch["obs"])
        else:
            inputs.append(batch['obs'][:, t])

        if self.args.obs_last_action:
            if t is None:
                first_step = th.zeros((batch['actions'].shape[0], 1, *batch['actions'].shape[2:]), device=batch.device)
                others_steps = batch['actions'][:, 1:]
                last_actions = th.concat([first_step, others_steps], dim=1)
                inputs.append(last_actions)
            else:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t-1])

        if self.args.obs_agent_id:
            if t is None:
                num_steps = batch['state'].shape[1]
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(1).expand(bs, num_steps, -1, -1))
            else:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if t is None:
            num_steps = batch['state'].shape[1]
            inputs = th.cat([x.reshape(bs*self.n_agents*num_steps, -1) for x in inputs], dim=1)
        else:
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
