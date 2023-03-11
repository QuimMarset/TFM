from gym import spaces
import torch as th
import numpy as np
from .non_shared_controller import NonSharedMAC



class CQMixNonSharedMAC(NonSharedMAC):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore_agent_ids=None):

        chosen_actions = self.forward(ep_batch[bs], t_ep)
        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")

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


    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs = []
        hidden_states = []

        for i, agent in enumerate(self.agents):
            actions_i, hidden_states_i = agent(agent_inputs[:, i], self.hidden_states[i])
            agent_outs.append(actions_i)
            hidden_states.append(hidden_states_i)

        self.hidden_states = hidden_states
        agent_outs = th.stack(agent_outs).squeeze(1)
        
        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                    + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]

        return input_shape
