from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np
from .shared_but_sides_controller import SharedButSidesMAC


# This multi-agent controller has 3 agents (pettingzoo pistonball)
class CQMixMACSides(SharedButSidesMAC):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore_agent_ids=None):

        chosen_actions = self.forward(ep_batch[bs], t_ep)
        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        
        # Now do appropriate noising
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        # Ornstein-Uhlenbeck:
        if not test_mode:  # do exploration
            
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)

                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args["episode_limit"]*self.args.ou_stop_episode else 0.0
                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                end_noise = getattr(self.args, "end_noise", 10000000000)
                act_noise = getattr(self.args, "act_noise", 0)
                if t_env >= start_steps:
                    if t_env <= end_noise:
                        if explore_agent_ids is None:
                            x = chosen_actions.clone().zero_()
                            chosen_actions += act_noise * x.clone().normal_()
                        else:
                            for idx in explore_agent_ids:
                                x = chosen_actions[:, idx].clone().zero_()
                                chosen_actions[:, idx] += act_noise * x.clone().normal_()
                else:
                    if self.args.env_args.get("scenario", None) is not None and self.args.env_args["scenario"] in ["Humanoid-v4", "HumanoidStandup-v4"]:
                        chosen_actions = th.from_numpy(np.array([[self.args.action_spaces[0].sample() for i in range(self.n_agents)] 
                            for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)
                    else:
                        chosen_actions = th.from_numpy(np.array([[self.args.action_spaces[i].sample() for i in range(self.n_agents)] 
                            for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)

        # For continuous actions, now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].low[_actid].item(),
                                                           self.args.action_spaces[_aid].high[_actid].item())
        
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):   
            # NOTE: This was added to handle scenarios like simple_reference since action space is Tuple
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                                                           self.args.action_spaces[_aid].spaces[0].high[_actid])
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                                                            self.args.action_spaces[_aid].spaces[1].high[_actid])
        return chosen_actions


    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()


    def forward(self, ep_batch, t):
        shared_inputs, side_1_inputs, side_2_inputs = self._build_inputs(ep_batch, t)

        shared_outs, self.shared_hidden_states = self.shared_agent(shared_inputs, self.shared_hidden_states)
        side_1_outs, self.side_1_hidden_states = self.side_agent_1(side_1_inputs, self.side_1_hidden_states)
        side_2_outs, self.side_2_hidden_states = self.side_agent_2(side_2_inputs, self.side_2_hidden_states)

        if ep_batch.batch_size > 1:
            shared_outs = shared_outs.reshape((ep_batch.batch_size, self.n_shared, -1))
            side_1_outs = side_1_outs.unsqueeze(dim=1)
            side_2_outs = side_2_outs.unsqueeze(dim=1)
            agent_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=1)
            agent_outs = agent_outs.view(ep_batch.batch_size * self.n_agents, -1)
        else:
            agent_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=0)

        return agent_outs


    def build_inputs_common(self, batch, t, start_index, end_index, is_shared):
        batch_size = batch.batch_size
        inputs = []
        inputs.append(batch['obs'][:, t, start_index:end_index])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch['actions'][:, t, start_index:end_index]))
            else:
                inputs.append(batch['actions'][:, t-1, start_index:end_index])

        if self.args.obs_agent_id and is_shared:
            agent_ids = th.eye(self.n_shared, device=batch.device).unsqueeze(0).expand(batch_size, -1, -1)
            inputs.append(agent_ids)

        if is_shared:
            reshape_dim = [batch_size * self.n_shared, -1]
        else:
            reshape_dim = [batch_size, -1]
        inputs = th.cat([x.reshape(*reshape_dim) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        
        if self.args.obs_agent_id:
            input_shape += self.n_shared

        return input_shape