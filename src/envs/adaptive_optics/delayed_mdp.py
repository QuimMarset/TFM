from collections import deque


class DelayedMDP:
    
    def __init__(self, delay):
        self.delay = delay
        # for reward assignment
        self.action_list = deque(maxlen=delay+1)
        self.obs_list = deque(maxlen=delay+1)
        self.state_list = deque(maxlen=delay+1)
        self.next_state_list = deque(maxlen=delay+1)
        self.next_obs_list = deque(maxlen=delay+1)
        self.next_action_list = deque(maxlen=delay+1)


    def check_update_possibility(self):
        """
        Checks that action list (and all the lists by the same rule)
        have enough information to take into account the delay
        """
        return len(self.action_list) >= (self.delay + 1)


    def save(self, state, obs, action, next_state, next_obs):
        self.action_list.append(action)
        self.state_list.append(state)
        self.obs_list.append(obs)
        self.next_state_list.append(next_state)
        self.next_obs_list.append(next_obs)


    def credit_assignment(self):
        return self.state_list[0], self.obs_list[0], self.action_list[0], \
            self.next_state_list[-1], self.next_obs_list[-1]