from controllers.base_controller import BaseController
from components.action_noising import GaussianNoise, ActionClamper, ActionSampler



class DDPGController(BaseController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_shape = args.action_shape
        self.action_noising = GaussianNoise(args.sigma_start, args.sigma_finish, 
                                            args.sigma_anneal_time + args.start_steps, 
                                            args.decay_type, args.power)
        self.action_clamper = ActionClamper(args.action_spaces, args.device)
        self.action_sampler = ActionSampler(args.action_spaces)
        self.init_hidden(args.batch_size)

    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # (b, n_agents, action_shape)
        actions = self.forward(ep_batch, t_ep)
        if not test_mode:
            if t_env <= self.args.start_steps:
                actions = self.action_sampler.sample_actions(actions.size(0), actions.device)
            else:
                actions = self.action_noising.add_noise(actions, t_env)
                actions = self.action_clamper.clamp_actions(actions)      
        return actions[bs]
    

    def select_actions_train(self, ep_batch, t_ep):
        # (b, n_agents, action_shape)
        actions = self.forward(ep_batch, t_ep)
        if self.args.env != 'adaptive_optics':
            # (b, 1, action_shape * n_agents)
            return actions.view(ep_batch.batch_size, 1, self.n_agents * self.action_shape)
        return actions


    def forward(self, ep_batch, t):
        # (b, 1, state_shape)
        agent_inputs = self._build_inputs(ep_batch, t)
        # (b, 1, action_shape * n_agents), (b, 1, hidden_dim)
        actions, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # (b, n_agents, action_shape)
        if self.args.env != 'adaptive_optics':
            return actions.view(ep_batch.batch_size, self.n_agents, self.action_shape)
        return actions


    def _build_inputs(self, batch, t):
        # State has dimensions: (b, 1, state_shape)
        return batch["state"][:, t].unsqueeze(1)


    def _get_input_shape(self, scheme):
        return scheme["state"]["vshape"]
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)