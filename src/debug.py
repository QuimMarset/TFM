from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
from pettingzoo.mpe.simple_tag_v2 import parallel_env
import matplotlib.pyplot as plt
from PIL import Image
import gym




if __name__ == '__main__':

    kwargs = {
        'episode_limit': 1000,
        'scenario': "manyagent_swimmer",
        'agent_conf': "4x1",
        'agent_obsk': 0,
        'obs_add_global_pos': False,
        'render_mode': "human"
    }

    env = MujocoMulti(**kwargs)
    env.reset()
    #initial_state = env.render()

    action_spaces = env.get_action_spaces()
    env.get_obs()


    for i in range(1000):
        env.render()

        actions = []
        for action_space in action_spaces:
            action = action_space.sample()
            actions.append(action)
        rewards, terminated, _ = env.step(actions)
        
        if terminated:
            env.reset()
