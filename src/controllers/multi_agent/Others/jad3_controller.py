from controllers.multi_agent.Others.facmac_controller import FACMACAgentController
from components.action_noising import GaussianNoise, OrnsteinUhlenbeckNoise, NoiseClamper
import torch as th



class JAD3Controller(FACMACAgentController):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self._create_target_noise(args)

            
    def select_target_actions(self, ep_batch, t_ep, t_env):
        actions = self.forward(ep_batch, t_ep)
        noise = self.target_action_noising.generate_noise(actions, t_env)
        noise = self.noise_clamper.clamp_noise(noise)
        actions = self.action_clamper.clamp_actions(actions + noise)        
        return actions


    def _create_target_noise(self, args):
        if args.use_training_steps_to_compute_target_noise:
            start_steps = 0
        else:
            start_steps = args.start_steps

        target_anneal_time = args.target_sigma_anneal_time + start_steps

        self.noise_clamper = NoiseClamper(args.target_noise_clipping)
        
        self.target_action_noising = \
                GaussianNoise(args.target_sigma_start, args.target_sigma_finish, 
                              target_anneal_time, args.decay_type, args.power)
         
            