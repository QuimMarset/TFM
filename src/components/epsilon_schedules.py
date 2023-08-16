import numpy as np



class LinearDecaySchedule:

    def __init__(self, start, finish, time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length


    def eval(self, step):
        return max(self.finish, self.start - self.delta * step)
    


class ExponentialDecaySchedule:
        
    def __init__(self, start, finish, time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1


    def eval(self, step):
            return min(self.start, max(self.finish, np.exp(- step / self.exp_scaling)))



class PolynomialDecaySchedule:

    def __init__(self, start, finish, time_length, power=2):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.power = power


    def eval(self, step):
        step = min(step, self.time_length)
        dif = self.start - self.finish
        step_size = (1 - step / self.time_length) ** (self.power)
        return dif * step_size + self.finish
    


def create_decay_schedule(type, start, finish, time_length, power):
    if type == 'linear':
        return LinearDecaySchedule(start, finish, time_length)
    
    elif type == 'exp':
        return ExponentialDecaySchedule(start, finish, time_length)

    elif type == 'polynomial':
        return PolynomialDecaySchedule(start, finish, time_length, power)
    
    else:
        raise ValueError(f'Unknown decay schedule type {type}')
    