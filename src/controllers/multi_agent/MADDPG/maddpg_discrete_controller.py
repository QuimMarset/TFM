import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from controllers.base_controller import BaseController
from controllers.multi_agent.q_controller import QController



def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class MADDPGDiscreteController(QController):

    def __init__(self, scheme, args):
        BaseController.__init__(self, scheme, args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        logits = self.forward(ep_batch, t_ep)
        actions = gumbel_softmax(logits, hard=True).argmax(dim=-1)
        # (b, n_agents, 1)
        return actions.unsqueeze(-1)[bs]
    

    def select_target_actions(self, ep_batch, t_ep):
        logits = self.forward(ep_batch, t_ep)
        # (b, n_agents, n_discrete_actions)
        actions = onehot_from_logits(logits)
        return actions
    

    def select_actions_with_logits(self, ep_batch, t_ep):
        logits = self.forward(ep_batch, t_ep)
        # (b, n_agents, n_discrete_actions)
        actions = gumbel_softmax(logits, hard=True)
        return actions, logits


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        logits, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return logits.view(ep_batch.batch_size, self.n_agents, -1)
    