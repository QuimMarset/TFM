from controllers.multi_agent.FACMAC.facmac_controller_no_rnn import FACMACAgentControllerNoRNN
from components.action_noising import GaussianClampedNoise



class FACMACTD3AgentControllerNoRNN(FACMACAgentControllerNoRNN):

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.target_action_noising = GaussianClampedNoise(args.target_noise_clipping, 
                                                          args.target_sigma_start, args.target_sigma_finish, 
                                                          args.target_sigma_anneal_time, args.start_steps)


    def select_target_actions(self, ep_batch, t_ep, t_env):
        actions = self.forward(ep_batch, t_ep)
        actions = self.target_action_noising.add_noise(actions, t_env)
        actions = self.action_clamper.clamp_actions(actions)
        return actions
