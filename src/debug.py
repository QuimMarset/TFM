from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
import time


if __name__ == '__main__':

    env_args = {
        'episode_limit': 1000,
        'scenario': "Ant-v4",
        'agent_conf': "4x2",
        'agent_obsk': 0,
        'global_categories': "qvel",
        'healthy_reward': 0
    }

    env = MujocoMulti(**env_args)
    env.reset()

    state = env.get_state()
    obs = env.get_obs()
    action_spaces = env.get_action_spaces()

    for i in range(5000):

        actions = [action_space.sample() for action_space in action_spaces]

        env.render()

        reward, done, truncated, _  = env.step(actions)

        if done or truncated:
            print(f'Reset at {i}')
            env.reset()

        time.sleep(0.01)

    print()