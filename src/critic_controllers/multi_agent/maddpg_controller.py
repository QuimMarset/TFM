from critic_controllers.base_controller import BaseCriticController
import torch as th



class MADDPGCriticController(BaseCriticController):

    # Used with continuous actions

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.init_hidden(args.batch_size)
        self.action_shape = args.action_shape
        self.state_shape = scheme['state']['vshape']


    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]

        if self.args.critic_obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        
        return input_shape
    

    def _get_action_shape(self, scheme):
        if self.args.env == 'adaptive_optis':
            # state and full action have the same last dimensions (i.e. width and height)
            return scheme["state"]["vshape"]
        return scheme["actions"]["vshape"]


    def forward(self, ep_batch, t_ep, actions):
        inputs = self._build_inputs(ep_batch, t_ep)
        # (b, 1, 1), (b, hidden_dim)
        critic_outs, self.hidden_states = self.critic(inputs, self.hidden_states, actions)
        return critic_outs
    

    def _build_inputs(self, batch, t_ep):
        inputs = []
        states = batch['state'][:, t_ep]
        inputs.append(states)

        if self.args.critic_obs_individual_obs:
            obs = batch["obs"][:, t_ep].view(-1, self.n_agents * self.args.obs_shape)
            inputs.append(obs)

        inputs = th.cat(inputs, dim=-1)
        return inputs
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.critic.init_hidden()
        # (1, h) -> (b, h)
        self.hidden_states = self.hidden_states.expand(batch_size, -1)