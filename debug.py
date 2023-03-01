import gymnasium as gym
import mujoco
import numpy as np
from multiagent_mujoco.mujoco_multi import MujocoMulti
import time


env = gym.make('Ant-v4', render_mode="human")
env.reset()
print(env.action_space)
print(env.observation_space)

for i in range(1000):

    action = 2 * np.random.rand(8) - 1

    obs, reward, terminated, truncated, info = env.step(action)

#def main():
#    env_args = {"scenario": "Ant-v4",
#                "agent_conf": "8x1",
#                "agent_obsk": 0,
#                "episode_limit": 1000,
#                "render_mode": "human"
#                }
#    env = MujocoMulti(env_args=env_args)
#    env_info = env.get_env_info()
#
#    n_actions = env_info["n_actions"]
#    n_agents = env_info["n_agents"]
#    n_episodes = 10
#    print(n_agents)
#
#    for e in range(n_episodes):
#        env.reset()
#        terminated = False
#        episode_reward = 0
#
#        while not terminated:
#            obs = env.get_obs()
#            state = env.get_state()
#
#            actions = []
#            for agent_id in range(n_agents):
#                avail_actions = env.get_avail_agent_actions(agent_id)
#                avail_actions_ind = np.nonzero(avail_actions)[0]
#                action = np.random.uniform(-1.0, 1.0, n_actions)
#                actions.append(action)
#
#            reward, terminated, _ = env.step(actions)
#            episode_reward += reward
#
#            time.sleep(0.01)
#            env.render()
#
#
#        print("Total reward in episode {} = {}".format(e, episode_reward))
#
#    env.close()
#
#if __name__ == "__main__":
#    main()