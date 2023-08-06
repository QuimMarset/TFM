from controllers.single_agent.ddpg_controller import DDPGController
from components.action_noising import GaussianClampedNoise



class TD3Controller(DDPGController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.target_action_noising = GaussianClampedNoise(args.target_noise_clipping,
                                                          args.target_sigma_start, args.target_sigma_finish, 
                                                          args.target_sigma_anneal_time, args.start_steps)
    

    def select_target_actions(self, ep_batch, t_ep, t_env):
        # (b, n_agents, action_shape)
        actions = self.forward(ep_batch, t_ep)
        # Need to keep the dimension because the action space assumes multiple agents
        actions = self.target_action_noising.add_noise(actions, t_env)
        actions = self.action_clamper.clamp_actions(actions)
        # (b, 1, action_shape * n_agents)
        return actions.view(ep_batch.batch_size, 1, self.n_agents * self.action_shape)