from controllers.single_agent.ddpg_controller import DDPGController
from components.action_noising import GaussianNoise, NoiseClamper



class TD3Controller(DDPGController):


    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.noise_clamper = NoiseClamper(args.target_noise_clipping)

        if args.use_training_steps_to_compute_target_noise:
            start_steps = 0
        else:
            start_steps = args.start_steps

        self.target_action_noising = GaussianNoise(args.target_sigma_start, args.target_sigma_finish, 
                                                   args.target_sigma_anneal_time + start_steps,
                                                   args.decay_type, args.power)
    

    def select_target_actions(self, ep_batch, t_ep, t_env):
        # (b, n_agents, action_shape)
        actions = self.forward(ep_batch, t_ep)
        # Need to keep the dimension because the action space assumes multiple agents
        noise = self.target_action_noising.generate_noise(actions, t_env)
        noise = self.noise_clamper.clamp_noise(noise)
        actions = self.action_clamper.clamp_actions(actions + noise)
        # (b, 1, action_shape * n_agents)
        return actions.view(ep_batch.batch_size, 1, self.n_agents * self.action_shape)
        