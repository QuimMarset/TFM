from critic_controllers.base_controller import BaseCriticController
import torch as th



class MADDPGCriticController(BaseCriticController):

    # Used with continuous actions

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.init_hidden(args.batch_size)
        self.action_shape = args.action_shape


    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]

        if self.args.critic_obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        
        if self.args.critic_add_agent_id:
            input_shape += self.n_agents
        
        return input_shape
    

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"][0] * self.n_agents


    def forward(self, ep_batch, t_ep, actions):
        inputs = self._build_inputs(ep_batch, t_ep)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_inputs(self, batch, t_ep):
        inputs = []
        states = batch['state'][:, t_ep].unsqueeze(1).expand(-1, self.n_agents, -1)
        inputs.append(states)

        if self.args.critic_obs_individual_obs:
            inputs.append(batch["obs"][:, t_ep])

        if self.args.critic_add_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1)
            inputs.append(agent_ids)

        inputs = th.cat(inputs, dim=-1)
        return inputs
        