from abc import ABC
import gym
from collections import deque, OrderedDict
import numpy as np
from envs.adaptive_optics.shesha.supervisor.rlSupervisor import RlSupervisor as Supervisor
from envs.adaptive_optics.shesha.util.utilities import load_config_from_file
import math
from gym import spaces


class AoEnv(gym.Env, ABC):

    #            Initialization
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def __init__(self,
                 config_env_rl,
                 parameter_file,
                 seed,
                 device):

        super(AoEnv, self).__init__()

        # Config rl
        self.config_env_rl = config_env_rl

        # Loading Compass config
        fd_parameters = "envs/adaptive_optics/parameter_files/"
        config_compass = load_config_from_file(fd_parameters + parameter_file)

        self.supervisor = Supervisor(config=config_compass,
                                     n_reverse_filtered_from_cmat=self.config_env_rl['n_reverse_filtered_from_cmat'],
                                     include_tip_tilt=False,
                                     filter_commands=False,
                                     command_filter_value=500,
                                     initial_seed=seed,
                                     which_modal_basis="Btt",
                                     mode=self.config_env_rl['mode'],
                                     device=device)

        # From supervisor for easy access
        self.command_shape = self.supervisor.command_shape  # always in 1D
        self.action_1d_shape = self.supervisor.action_1d_shape
        self.action_2d_shape = self.supervisor.action_2d_shape
        self.mask_valid_actuators = self.supervisor.mask_valid_actuators
        # Initializy the history
        self.s_dm_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm'])
        self.s_dm_residual_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual'])
        self.s_dm_residual_rl_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_rl'])
        self.a_for_reward_history = deque(maxlen=self.config_env_rl['number_of_previous_a_for_reward'])
        # Mask
        self.mask_saturation = None
        # Observation/action space
        self.state_size_channel_0, self.observation_shape,\
            self.observation_space, self.action_space = self.define_state_action_space()
        # Normalization
        self.norm_parameters = {"dm":
                                    {"mean": 0.0,
                                     "std": self.config_env_rl['dm_std']},
                                "dm_residual":
                                            {"mean": 0.0,
                                             "std": self.config_env_rl['dm_residual_std']}
                                }
        # Delayed assignment
        self.delayed_assignment = self.config_env_rl['delayed_assignment']
        # Defined for the state
        self.s_next_main = None

        print("Parmeter file {} Observation space {} Action space {}".format(parameter_file, self.observation_space.shape, self.action_space.shape))

    #           Basic environment
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def define_state_action_space(self):

        state_size_channel_0 = int(self.config_env_rl['number_of_previous_s_dm']) +\
                               int(self.config_env_rl['number_of_previous_s_dm_residual']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_rl']) +\
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    # For history
    def append_to_attr(self, attr, value):
        self.__dict__[attr].append(value)

    def clear_attr(self, attr):
        self.__dict__[attr].clear()

    def reset_history(self, attributes_list):
        idx = 0
        for attr in attributes_list:
            attr_history = attr + "_history"
            shape_of_history = np.zeros(self.action_2d_shape, dtype="float32")
            self.clear_attr(attr_history)
            rang = self.config_env_rl["number_of_previous_" + attr]
            for _ in range(rang):
                self.append_to_attr(attr_history, shape_of_history)
            idx += 1

    def reset(self,
              only_reset_dm: bool = False,
              return_dict: bool = False,
              add_one_to_seed=True):

        #print("Resetting")
        if add_one_to_seed:
            self.supervisor.add_one_to_seed()
        self.supervisor.reset(only_reset_dm)
        self.reset_history(['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward'])
        if not only_reset_dm:
            self.supervisor.move_atmos_compute_wfs_reconstruction()

        self.build_state()
        s = self.get_next_state(return_dict=return_dict)
        return s

    def standardise(self, inpt, key):
        """
        standardises
        :param inpt: state to be normalized
        :param key: "wfs" or "dm"
        :return: input normalized
        """

        mean = self.norm_parameters[key]['mean']
        std = self.norm_parameters[key]['std']
        return (inpt - mean) / std

    def filter_actions(self, a):

        # 0) To 1D
        a = self.supervisor.apply_projector_volts2d_to_volts1d(a)

        # 1) In case of actuator space filter with Btt if necessary
        if self.config_env_rl['filter_state_actuator_space_with_btt']:
            a = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a,
                                                                               add_tip_tilt_to_not_break=True if self.supervisor.num_dm > 1 else False)

        return self.supervisor.apply_projector_volts1d_to_volts2d(a)

    def add_s_dm_info(self, s_next, s_dm_info, key_attr, key_norm):

        key_attribute_history = key_attr + "_history"
        current_history = getattr(self, key_attribute_history)
        for idx in range(len(current_history)):
            past_s_dm_info = getattr(self, key_attribute_history)[idx]
            past_s_dm_info = self.process_dm_state(past_s_dm_info, key=key_norm)
            s_next[key_attribute_history + "_" + str(len(current_history) - idx)] = \
                past_s_dm_info

        if self.config_env_rl["number_of_previous_" + key_attr] > 0:
            self.append_to_attr(key_attribute_history, s_dm_info)

        # 2 Add current residual to the state
        if self.config_env_rl[key_attr]:
            s_dm = self.process_dm_state(s_dm_info, key=key_norm)
            s_next[key_attr] = s_dm.copy()
        return s_next

    def process_dm_state(self, s_dm, key):
        if self.config_env_rl['normalization_bool']:
            s_dm = self.standardise(s_dm, key=key)
        return s_dm

    def calculate_linear_residual(self):
        c_linear = self.supervisor.rtc.get_err(0)
        if self.config_env_rl['filter_state_actuator_space_with_btt']:
            c_linear = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(c_linear)

        return c_linear

    def calculate_reward(self, a):

        r2d_linear_rec_pzt = np.square(self.preprocess_dm_info(self.calculate_linear_residual()))

        if self.config_env_rl['value_action_penalizer'] > 0 and a is not None:
            self.a_for_reward_history.append(a)
            r2d_linear_rec_pzt = r2d_linear_rec_pzt + \
                                 self.config_env_rl['value_action_penalizer'] * \
                                 np.square(self.a_for_reward_history[0])

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return -(r2d_linear_rec_pzt.mean())
        elif self.config_env_rl['reward_type'] == "log_scalar_actuators":
            return -np.log(r2d_linear_rec_pzt.mean())
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return -r2d_linear_rec_pzt
        else:
            raise NotImplementedError

    #
    #         Step methods
    # This works for both level = Correction and level = Gain
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def preprocess_dm_info(self, s_dm_info):
        if self.supervisor.num_dm == 2:
            s_pzt_info = s_dm_info[:-2]
        else:
            s_pzt_info = s_dm_info

        s_dm = self.supervisor.apply_projector_volts1d_to_volts2d(s_pzt_info)
        if self.mask_saturation is not None:
            s_dm *= self.mask_saturation

        return s_dm

    def build_state(self):
        s_next = OrderedDict()
        if self.config_env_rl['number_of_previous_s_dm'] > 0 or self.config_env_rl['s_dm']:
            s_dm = self.supervisor.apply_projector_volts1d_to_volts2d(self.supervisor.past_command_rl)
            s_next = self.add_s_dm_info(s_next, s_dm, key_attr="s_dm", key_norm="dm")

        if self.config_env_rl['s_dm_residual_rl']:
            past_a = self.supervisor.apply_projector_volts1d_to_volts2d(self.supervisor.past_action_rl)
            s_next = self.add_s_dm_info(s_next, past_a, key_attr="s_dm_residual_rl", key_norm="dm_residual")

        if self.config_env_rl['s_dm_residual']:
            s_dm_residual = self.preprocess_dm_info(self.calculate_linear_residual())
            # print("s_dm_residual", s_dm_residual[self.mask_valid_actuators==1][0])
            s_next = self.add_s_dm_info(s_next, s_dm_residual, key_attr="s_dm_residual", key_norm="dm_residual")

        self.s_next_main = s_next

    def get_next_state(self, return_dict):
        if return_dict:
            return self.s_next_main
        else:
            return np.stack(np.array(list(self.s_next_main.values())))

    def step(self, a, controller_type):
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a = self.supervisor.apply_projector_volts2d_to_volts1d(a)
            a = a * self.config_env_rl['action_scale']

        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a, controller_type=controller_type)
        r = self.calculate_reward(a)
        sr_se, sr_le, _, _ = self.supervisor.target.get_strehl(0)
        info = {"sr_se": sr_se,
                "sr_le": sr_le}
        # 3) Move atmos, compute WFS and reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()
        # 4) Build state
        self.build_state()
        s = self.get_next_state(return_dict=False)

        if self.supervisor.iter % (self.config_env_rl['reset_strehl_every_and_print'] + 1) == 0 and\
                self.supervisor.iter > 1:
            self.supervisor.target.reset_strehl(0)

        return s, r, False, info
