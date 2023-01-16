import matplotlib.pyplot as plt
from pettingzoo.butterfly.pistonball_v6_positions import parallel_env
from envs.pettingzoo_wrapper import PettingZooWrapper
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    name = 'pistonball_reward'
    pretrained = False
    params = {
        'num_pistons': 10,
        'continuous' : False,
        'render' : False,
        'seed' : 1234,
        'time_penalty': -0.1,
        'alpha' : 0.1
    }

    frames = []

    env = PettingZooWrapper(name, pretrained, **params)
    obs = env.reset()

    for i in range(20):

        actions = np.random.choice(2, 10)
        x = env.step(actions)

        frames.append(env.get_current_frame())
        
        obs = env.get_obs()
        state = env.get_state()
        for j in range(10):
            print(f'Agent {j}: {obs[j]}')

        print(state)

        print('\n\n')

    for i in range(len(frames)):
        plt.subplot(5, 5, i+1)
        plt.imshow(frames[i])
        plt.axis('off')

    plt.tight_layout()     
    plt.show()