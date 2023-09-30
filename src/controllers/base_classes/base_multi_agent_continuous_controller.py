import torch as th
from controllers.base_classes.base_controller import BaseController
from components.action_noising import GaussianNoise, ActionClamper, ActionSampler



class BaseMultiAgentContinuousController(BaseController):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_noising = GaussianNoise(args.sigma_start, args.sigma_finish, 
                                            args.sigma_anneal_time + args.start_steps, 
                                            args.decay_type, args.power)
        self.action_clamper = ActionClamper(args.action_spaces, args.device)
        self.action_sampler = ActionSampler(args.action_spaces)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = self.forward(ep_batch, t_ep)
        if not test_mode:
            actions = self._add_noise(actions, t_env)
        return actions[bs]


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        actions, self.hidden_states = self.agent(agent_inputs, self.hidden_states)                
        return actions


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.add_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])

        if self.args.add_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.add_last_action:
            input_shape += scheme["actions"]["vshape"]
        if self.args.add_agent_id:
            input_shape += self.n_agents
        return input_shape
    

    def _add_noise(self, actions, t_env):
        if self.args.start_steps > 0 and t_env <= self.args.start_steps:
            return self.action_sampler.sample_actions(actions.size(0), actions.device)
        else:
            noise = self.action_noising.generate_noise(actions, t_env)
            return self.action_clamper.clamp_actions(actions + noise)