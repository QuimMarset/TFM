from critic_controllers.multi_agent.factorized_controller import FactorizedCriticController
import torch as th



class FactorizedCriticControllerNoRNN(FactorizedCriticController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.init_hidden(args.batch_size)


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.critic_use_previous_transitions and self.args.num_previous_transitions > 0:
            input_shape += scheme["obs"]["vshape"] * self.args.num_previous_transitions
            input_shape += scheme["actions"]["vshape"] * self.args.num_previous_transitions
        
        if self.args.critic_add_agent_id:
            input_shape += self.n_agents
        
        return input_shape
        

    def _get_action_shape(self, scheme):
        return scheme["actions"]["vshape"]
     

    def forward(self, ep_batch, step, actions):
        inputs = self._build_inputs(ep_batch, step)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs
    

    def _build_inputs(self, batch, step):
        inputs = []
        inputs.append(batch['obs'][:, step])

        if self.args.critic_use_previous_transitions and self.args.num_previous_transitions > 0:
            inputs.extend(self._get_previous_transitions(batch, step))

        if self.args.critic_add_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1)
            inputs.append(agent_ids)

        inputs = th.cat(inputs, dim=-1)
        return inputs
        

    def _get_previous_transitions(self, batch, step):
        inputs = []
        inputs.append(batch['prev_obs'][:, step])
        inputs.append(batch['prev_actions'][:, step])
        return inputs
        