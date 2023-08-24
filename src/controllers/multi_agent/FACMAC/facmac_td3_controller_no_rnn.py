from controllers.multi_agent.FACMAC.facmac_controller_no_rnn import FACMACAgentControllerNoRNN
from components.action_noising import GaussianNoise, OrnsteinUhlenbeckNoise, NoiseClamper
import torch as th



class FACMACTD3AgentControllerNoRNN(FACMACAgentControllerNoRNN):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self._create_target_noise(args)
        self._create_noise(args)

            
    def select_target_actions(self, ep_batch, t_ep, t_env):
        actions = self.forward(ep_batch, t_ep)
        noise = self.target_action_noising.generate_noise(actions, t_env)
        noise = self.noise_clamper.clamp_noise(noise)
        actions = self.action_clamper.clamp_actions(actions + noise)
        
        if self.args.env == 'adaptive_optics':
                actions *= th.tensor(self.args.agent_masks, device=actions.device, 
                                     requires_grad=False, dtype=th.float32)
        
        return actions


    def _create_target_noise(self, args):
        if args.use_training_steps_to_compute_target_noise:
            start_steps = 0
        else:
            start_steps = args.start_steps

        target_anneal_time = args.target_sigma_anneal_time + start_steps

        self.noise_clamper = NoiseClamper(args.target_noise_clipping)
        
        if args.use_ornstein:
            self.target_action_noising = \
                OrnsteinUhlenbeckNoise(args.ou_theta, args.ou_sigma, 
                                       args.ou_noise_scale, 
                                       target_anneal_time, 
                                       args.decay_type, args.power)

        else:
            self.target_action_noising = \
                GaussianNoise(args.target_sigma_start, args.target_sigma_finish, 
                              target_anneal_time, args.decay_type, args.power)
         

    def _create_noise(self, args):
        if args.use_ornstein:
            self.action_noising = \
                OrnsteinUhlenbeckNoise(args.ou_theta, args.ou_sigma, 
                                       args.ou_noise_scale, 
                                       args.sigma_anneal_time + args.start_steps, 
                                       args.decay_type, args.power)
            