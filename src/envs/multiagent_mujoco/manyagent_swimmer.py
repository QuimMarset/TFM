import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from jinja2 import Template
from gym.spaces import Box



class ManyAgentSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        agent_conf = kwargs.get("agent_conf")
        
        self.n_agents = int(agent_conf.split("x")[0])
        self.n_segments_per_agents = int(agent_conf.split("x")[1])
        # Number of controllable hinges
        self.n_controllable_segments = self.n_agents * self.n_segments_per_agents
        self.n_segments = self.n_controllable_segments + 1

        self.forward_reward_weight = kwargs.get('forward_reward_weight', 1.0)
        self.ctrl_cost_weight = kwargs.get('ctrl_cost_weight', 0.0001)
        self.render_mode = kwargs.get('render_mode', None)

        self.exclude_current_positions_from_observation = \
            kwargs.get('exclude_current_positions_from_observation', True)
        
        self.state_entity_mode = kwargs.get('state_entity_mode', False)

        self._set_asset_path()
        self._create_xml_file()

        # Single-agent observation space
        observation_space = Box(-np.inf, np.inf, (self.n_controllable_segments * 2,))
        
        self.frame_skip = 4

        mujoco_env.MujocoEnv.__init__(self, self.asset_path, self.frame_skip, observation_space, self.render_mode)
        utils.EzPickle.__init__(self)

        self._set_entity_attributes()        

    
    def _set_asset_path(self):
        module_absolute_path = os.path.dirname(os.path.abspath(__file__))
        file_name = f'manyagent_swimmer_{self.n_agents}_agents_each_{self.n_segments_per_agents}_segments.auto.xml'
        self.asset_path = os.path.join(module_absolute_path, 'assets', file_name) 

    
    def _create_xml_file(self):
        if not os.path.exists(self.asset_path):
            print(f"Auto-Generating Manyagent Swimmer asset with {self.n_controllable_segments} controllable segments at {self.asset_path}.")
            self._generate_asset(self.n_controllable_segments, self.asset_path)


    def _set_entity_attributes(self):
        # The number of hinges each agent controls
        self.n_entities_obs = self.n_segments_per_agents
        # The number of total segments (i.e. controllable + 1)
        self.n_entities_state = self.n_segments
        # Observation only contains the angle and angular velocity of each controllable hinge
        self.obs_entity_feats = 2
        # State contains the features of all the joints defined per each segment. 
        # The first has the hinge plus the sliders
        self.state_entity_feats = 4 if self.exclude_current_positions_from_observation else 6


    def _generate_asset(self, n_controllable_segments, asset_path):
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                     'manyagent_swimmer.xml.template')
        
        with open(template_path, "r") as f:
            t = Template(f.read())
        
        body_str_template = """
        <body name="mid{:d}" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 {:d}" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        body_end_str_template = """
        <body name="back" pos="-1 0 0">
            <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
          </body>
        """

        body_close_str_template ="</body>\n"
        actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

        body_str = ""
        
        for i in range(1, n_controllable_segments - 1):
            body_str += body_str_template.format(i, (-1)**(i+1), i)
        
        body_str += body_end_str_template.format(n_controllable_segments - 1)
        body_str += body_close_str_template * (n_controllable_segments - 2)

        actuator_str = ""
        for i in range(n_controllable_segments):
            actuator_str += actuator_str_template.format(i)

        rt = t.render(body=body_str, actuators=actuator_str)

        if not os.path.exists(asset_path):
            with open(asset_path, "w") as f:
                f.write(rt)


    def control_cost(self, action):
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        return control_cost


    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        obs = self._get_obs()

        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
       
        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, info
    

    def _get_obs(self):
        if self.state_entity_mode:
            return self._get_obs_entity()
        return self._get_obs_default()
    

    def _get_obs_default(self):
        # Used as state rather than observation
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self.exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation


    def _get_obs_entity(self):
        """
        Used as state rather than observation
        Defined as the composition of the segments' joint features
        First segment has extra features because of the extra slider joints
        pos_x and pos_y not included if exclude_current_positions_from_observation = True
        Hence, we only consider 4 features
        [
            [angle_x_axis, deriv_angle, vel_x, vel_y, pos_x, pos_y]
            [angle_x_axis, deriv_angle, 0, 0, 0, 0] * n_segments
        ]
        """
        num_features = 6
        if self.exclude_current_positions_from_observation:
            num_features = 4
        
        obs = np.zeros((self.n_segments, num_features))

        obs[0, 0] = self.data.qpos[2]
        obs[0, 1] = self.data.qvel[2]
        obs[0, 2:4] = self.data.qvel[:2]

        if not self.exclude_current_positions_from_observation:
            obs[0, 4:] = self.data.qpos[:2]

        for i in range(self.n_segments - 1):
            obs[i, 0] = self.data.qpos[i + 3]
            obs[i, 1] = self.data.qvel[i + 3]

        return obs.flatten()


    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
    

    def render(self, **kwargs):
        return super().render()
