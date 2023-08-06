from components.epsilon_schedules import DecayThenFlatSchedule
import torch as th
import numpy as np



class ActionClamper:

    def __init__(self, action_spaces, device):
        self.low_actions  = th.cat([th.tensor(action_space.low).view(1, -1) 
                                    for action_space in action_spaces], dim=0).to(device)
        self.high_actions = th.cat([th.tensor(action_space.high).view(1, -1) 
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

    def __init__(self, sigma_start, sigma_finish, sigma_anneal_time, start_steps):
        sigma_anneal_time = sigma_anneal_time + start_steps
        self.schedule = DecayThenFlatSchedule(sigma_start, sigma_finish, sigma_anneal_time, decay="linear")
        self.sigma = self.schedule.eval(0)
          

    def add_noise(self, agent_inputs, t_env):
        self.sigma = self.schedule.eval(t_env)
        noise = th.randn_like(agent_inputs) * self.sigma
        actions = agent_inputs + noise
        return actions
    


class GaussianClampedNoise(GaussianNoise):


    def __init__(self, noise_clipping, sigma_start, sigma_finish, sigma_anneal_time, start_steps):
        super().__init__(sigma_start, sigma_finish, sigma_anneal_time, start_steps)
        self.noise_clipping = noise_clipping


    def add_noise(self, agent_inputs, t_env):
        self.sigma = self.schedule.eval(t_env)
        noise = th.randn_like(agent_inputs) * self.sigma
        clamped_noise = th.clamp(noise, -self.noise_clipping, self.noise_clipping)
        actions = agent_inputs + clamped_noise
        return actions
    