import torch.nn as nn
import torch as th
import torch.nn.functional as F


class MLPContinuousStochasticActor(nn.Module):

    def __init__(self, input_shape, args):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # n_actions is the action shape
        self.mean = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.log_std = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def init_hidden(self):
        return None
    

    def forward(self, inputs):
        relu = th.nn.ReLU(inplace=False)
        fc1_out = relu(self.fc1(inputs))
        fc2_out = relu(self.fc2(fc1_out))
        means = self.mean(fc2_out)
        log_stds = self.log_std(fc2_out)
        return means, log_stds
