import torch as th
import numpy as np
from .basic_controller import BasicMAC


class FACMACAgentController(BasicMAC):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.ou_noise_state = None


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = self.forward(ep_batch, t_ep)
        if not test_mode:
            actions = self._add_noise(actions, t_env)
            actions = self._clamp_actions(actions)
        return actions[bs]


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        actions, self.hidden_states = self.agent(agent_inputs, self.hidden_states)                
        return actions.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
    

    def _add_noise(self, actions, t_env):
        if t_env >= self.args.stop_noise_step:
            return actions
        elif t_env <= self.args.start_steps:
            return self._sample_actions(actions)
        elif self.args.exploration_mode == 'gaussian':
            return self._add_gaussian_noise(actions)
        return self._add_ornstein_uhlenbeck_noise(actions, t_env)


    def _add_gaussian_noise(self, actions):
        x = actions.clone().zero_()
        actions += self.args.act_noise * x.clone().normal_()
        return actions
    

    def _add_ornsein_uhlenbeck_noise(self, actions):
        if self.ou_noise_state is None:
            self.ou_noise_state = actions.clone().zero_()
        x = self.ou_noise_state
        mu = 0
        dx = self.args.ou_theta * (mu - x) + self.args.ou_sigma  * x.clone().normal_()
        self.ou_noise_state = x + dx
        ou_noise = self.ou_noise_state * self.args.ou_noise_scale
        return actions + ou_noise


    def _clamp_actions(self, actions):
        for index in range(self.n_agents):
            action_space = self.args.action_spaces[index]
            for dimension_num in range(action_space.shape[0]):
                min_action = action_space.low[dimension_num].item()
                max_action = action_space.high[dimension_num].item()
                actions[:, index, dimension_num].clamp_(min_action, max_action)
        return actions
    

    def _sample_actions(self, actions):
        batch_size = actions.size(0)
        sampled_actions = []
        for _ in range(batch_size):
            actions_temp = []
            for i in range(self.n_agents):
                actions_temp.append(self.args.action_spaces[i].sample())
            sampled_actions.append(actions_temp)
        sampled_actions = th.from_numpy(np.array(sampled_actions))
        return sampled_actions.float().to(device=actions.device)
