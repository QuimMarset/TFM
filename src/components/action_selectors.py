import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import LinearDecaySchedule
REGISTRY = {}



class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = LinearDecaySchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time)
        self.epsilon = self.schedule.eval(0)


    def select_action(self, agent_inputs, t_env, test_mode=False):
        # multi-agent -> agent_inputs: (b, n_agents, n_discrete_actions)
        # dqn -> agent_inputs: (b, 1, n_discrete_actions ** n_agents)

        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()

        uniform_distrib = Categorical(th.ones_like(agent_inputs))
        # (b, n_agents, 1)
        random_actions = uniform_distrib.sample().long()

        # (b, n_agents) or (b, 1)
        picked_actions = pick_random * random_actions + (1 - pick_random) * agent_inputs.max(dim=2)[1]
        return picked_actions



REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
