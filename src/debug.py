from envs.multiagent_mujoco.mujoco_multi import MujocoMulti
import time
import numpy as np


def quaternion_to_angle_axis(x, y, z, w):
    quaternion = [x, y, z, w]
    # Normalize the quaternion
    quaternion /= np.linalg.norm(quaternion)

    # Extract the scalar and vector parts
    w = quaternion[3]
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]

    # Calculate the rotation angle
    theta = 2 * np.arccos(w)

    # Calculate the rotation axis
    axis_norm = np.sqrt(1 - w**2)
    ax = x / axis_norm
    ay = y / axis_norm
    az = z / axis_norm

    return theta, np.array([ax, ay, az])



if __name__ == '__main__':

    env_args = {
        'episode_limit': 1000,
        'scenario': "Ant-v4",
        'agent_conf': "2x4",
        'agent_obsk': 0,
        'global_categories': "qpos,qvel",
        'render_mode': 'human',
        'healthy_reward': 0,
        'healthy_z_range': (-np.inf, np.inf)
    }

    env = MujocoMulti(**env_args)
    env.reset()

    first_state = env.get_state()
    obs = env.get_obs()
    action_spaces = env.get_action_spaces()

    for i in range(5000):

        actions = [action_space.sample() for action_space in action_spaces]

        env.render()

        reward, done, truncated, _  = env.step(actions)

        state = env.get_state()
        angle, axis = quaternion_to_angle_axis(state[1], state[2], state[3], state[4])
        print(state[0:5], angle, axis)

        if state[0] >= 1 or done or truncated:
            print(f'Reset at {i}')
            env.reset()

        time.sleep(0.1)

    print()