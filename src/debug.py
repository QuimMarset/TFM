from envs.multiagent_mujoco.ant_multi_direction import AntMultiDirectionMultiAgentEnv
import time


if __name__ == '__main__':

    env_args = {
        'episode_limit': 1000,
        'scenario': "Ant-v4",
        'agent_conf': "2x4",
        'agent_obsk': 0,
        'global_categories': "qvel",
        'render_mode' : 'human'
    }

    env = AntMultiDirectionMultiAgentEnv(**env_args)
    env.reset()
    print(f'Random direction: {env.random_direction}')

    state = env.get_state()
    obs = env.get_obs()
    action_spaces = env.get_action_spaces()

    for i in range(5000):

        actions = [action_space.sample() for action_space in action_spaces]
        actions[1][:] = 0

        reward, done, truncated, _  = env.step(actions)

        if done or truncated:
            print(f'Reset at {i}')
            env.reset()
            print(f'Random direction: {env.random_direction}')

        time.sleep(0.3)

    print()