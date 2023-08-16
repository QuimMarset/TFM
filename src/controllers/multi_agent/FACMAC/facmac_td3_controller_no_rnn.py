from controllers.multi_agent.FACMAC.facmac_controller_no_rnn import FACMACAgentControllerNoRNN
from components.action_noising import GaussianClampedNoise, GaussianClampedDecayNoise, OrnsteinUhlenbeckNoise
import torch as th



class FACMACTD3AgentControllerNoRNN(FACMACAgentControllerNoRNN):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)

        if args.use_training_steps_to_compute_target_noise:
            start_steps = 0
        else:
            start_steps = args.start_steps

        if args.use_ornstein:
            self.action_noising = OrnsteinUhlenbeckNoise(args.ou_theta, args.ou_sigma, 
                                                     args.ou_noise_scale, 
                                                     args.sigma_anneal_time + args.start_steps, 
                                                     args.decay_type, args.power)

        target_anneal_time = args.target_sigma_anneal_time + start_steps

        if args.decay_clipped_noise:
            self.target_action_noising = GaussianClampedDecayNoise(args.target_noise_clipping, args.clip_noise_end,
                                                                   args.target_sigma_start, args.target_sigma_finish, 
                                                                   target_anneal_time, args.decay_type, args.power)
        else:
            self.target_action_noising = GaussianClampedNoise(args.target_noise_clipping, 
                                                              args.target_sigma_start, args.target_sigma_finish, 
                                                              target_anneal_time, args.decay_type, args.power)
            
        if args.clip_exploration_actions:
            sigma_anneal_time = args.sigma_anneal_time + start_steps
            if args.decay_clipped_noise:
                self.action_noising = GaussianClampedDecayNoise(args.noise_clipping, args.clip_noise_end,
                                                                   args.sigma_start, args.sigma_finish, 
                                                                   sigma_anneal_time, args.decay_type, 
                                                                   args.power)
            else:
                self.action_noising = GaussianClampedNoise(args.noise_clipping, args.sigma_start, 
                                                           args.sigma_finish, sigma_anneal_time, 
                                                           args.decay_type, args.power)
        

    def select_target_actions(self, ep_batch, t_ep, t_env):
        actions = self.forward(ep_batch, t_ep)
        actions = self.target_action_noising.add_noise(actions, t_env)
        actions = self.action_clamper.clamp_actions(actions)
        if self.args.env == 'adaptive_optics':
                actions *= th.tensor(self.args.agent_masks, device=actions.device, requires_grad=False, dtype=th.float32)
        return actions
