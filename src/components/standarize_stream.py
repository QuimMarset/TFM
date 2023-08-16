import torch
from typing import Tuple



# Taken from: https://github.com/semitable/fast-marl

class RunningMeanStd(object):
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=False)
        self.var = torch.ones(shape, dtype=torch.float32, device=device, requires_grad=False)
        self.count = epsilon


    def update(self, arr):
        batch_mean = torch.mean(arr, dim=(0, 1))
        batch_var = torch.var(arr, dim=(0, 1))
        batch_count = arr.shape[0] * arr.shape[1]
        self.update_from_moments(batch_mean, batch_var, batch_count)


    def update_from_moments(self, batch_mean, batch_var, batch_count):
        with torch.no_grad():
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

            self.mean += delta * (batch_count / tot_count)
            self.var = m_2 / tot_count
            self.count = tot_count
            