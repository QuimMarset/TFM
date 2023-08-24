from components.epsilon_schedules import create_decay_schedule
import torch as th
import numpy as np



class ActionClamper:

    def __init__(self, action_spaces, device):
        self.low_actions  = th.cat([th.tensor(action_space.low).unsqueeze(0) 
                                    for action_space in action_spaces], dim=0).to(device)
        self.high_actions = th.cat([th.tensor(action_space.high).unsqueeze(0)
                                    for action_space in action_spaces], dim=0).to(device)
        

    def clamp_actions(self, actions):
        return th.clamp(actions, self.low_actions, self.high_actions)
    


class ActionSampler:

    def __init__(self, action_spaces):
        self.action_spaces = action_spaces

    
    def sample_actions(self, batch_size, device):
        sampled_actions = []
        
        for _ in range(batch_size):
            actions_temp = []
            for action_space in self.action_spaces:
                actions_temp.append(action_space.sample())
            sampled_actions.append(actions_temp)
        
        sampled_actions = th.from_numpy(np.array(sampled_actions))
        return sampled_actions.float().to(device=device)



class GaussianNoise:

    def __init__(self, sigma_start, sigma_finish, sigma_anneal_time, schedule_type, power=1):
        self.schedule = create_decay_schedule(schedule_type, sigma_start, sigma_finish, sigma_anneal_time, power)


    def generate_noise(self, agent_inputs, t_env):
        self.sigma = self.schedule.eval(t_env)
        noise = th.randn_like(agent_inputs) * self.sigma
        return noise
    

    
class OrnsteinUhlenbeckNoise:

    def __init__(self, theta, sigma, noise_scale_start, noise_scale_anneal_time, schedule_type, power=1):
        self.theta = theta
        self.sigma = sigma
        self.schedule = create_decay_schedule(schedule_type, noise_scale_start, 0, noise_scale_anneal_time, power)


    def generate_noise(self, agent_inputs, t_env):
        temp = getattr(self, 'noise_state', agent_inputs.clone().zero_())

        derivative = self.theta * - temp + self.sigma * temp.clone().normal_()
        self.noise_state = temp + derivative
        
        noise_scale = self.schedule.eval(t_env)
        noise = self.noise_state * noise_scale
        return noise



class NoiseClamper:

    def __init__(self, noise_clipping):
        self.noise_clipping = noise_clipping


    def clamp_noise(self, noise):
        return th.clamp(noise, -self.noise_clipping, self.noise_clipping)
    


class NoiseClamperWithDecay:

    def __init__(self, clipping_start, clipping_finish, anneal_time, schedule_type, power=1):
        self.clipping_schedule = create_decay_schedule(schedule_type, clipping_start, clipping_finish, anneal_time, power)


    def clamp_noise(self, noise, t_env):
        self.clipped_noise = self.noise_clipping_schedule.eval(t_env)
        return th.clamp(noise, -self.clipped_noise, self.clipped_noise)  
