


class Hinge:

    def __init__(self, name, angle_index, angular_vel_index, hinge_index):
        self.name = name
        self.angle_index = angle_index
        self.angular_vel_index = angular_vel_index
        self.hinge_index = hinge_index


class Torso:

    def __init__(self, position_start, position_end, orientation_start, orientation_end, velocity_start, 
                 velocity_end, angular_vel_start, angular_vel_end):
        self.position_start = position_start
        self.position_end = position_end
        self.orientation_start = orientation_start
        self.orientation_end = orientation_end
        self.velocity_start = velocity_start
        self.velocity_end = velocity_end
        self.angular_vel_start = angular_vel_start
        self.angular_vel_end = angular_vel_end


"""
    Consider yourself looking at the ant from a zenital view, 
    the legs are named as follow (the numbers correspong to hip and ankle index):
        BackLeft (3)  | BackRight (4)
        FrontLeft (1) | FrontRight (2)
"""


hip_1 = Hinge('hip_1', 7, 21, 0)
ankle_1 = Hinge('ankle_1', 8, 22, 1)

hip_3 = Hinge('hip_3', 9, 23, 2)
ankle_3 = Hinge('ankle_3', 10, 24, 3)

hip_4 = Hinge('hip_4', 11, 25, 4)
ankle_4 = Hinge('ankle_4', 12, 26, 5)

hip_2 = Hinge('hip_2', 13, 27, 6)
ankle_2 = Hinge('ankle_2', 14, 28, 7)


torso = Torso(0, 2, 3, 6, 15, 17, 18, 20)


def create_ant_partition(num_agents, partition=None):
    if num_agents == 1:
        return [(hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4)]
    elif num_agents == 2:
        return [(hip_1, ankle_1, hip_2, ankle_2), (hip_3, ankle_3, hip_4, ankle_4)]
    else:
        raise NotImplementedError('Pending to implement PyBullet Ant 4 agent partitions')



"""
    | Num | Observation                                                  | Name (in corresponding XML file)       | Joint |
    |-----|--------------------------------------------------------------|----------------------------------------|-------|
    | 0   | x-coordinate of the torso (centre)                           | torso                                  | free  |
    | 1   | y-coordinate of the torso (centre)                           | torso                                  | free  |
    | 2   | z-coordinate of the torso (centre)                           | torso                                  | free  |
    | 3   | x-orientation of the torso (centre)                          | torso                                  | free  |
    | 4   | y-orientation of the torso (centre)                          | torso                                  | free  |
    | 5   | z-orientation of the torso (centre)                          | torso                                  | free  |
    | 6   | w-orientation of the torso (centre)                          | torso                                  | free  |
    | 7   | angle between torso and first link on front left             | hip_1 (front_left_leg)                 | hinge |
    | 8   | angle between the two links on the front left                | ankle_1 (front_left_leg)               | hinge |
    | 9   | angle between torso and first link on front right            | hip_2 (front_right_leg)                | hinge |
    | 10  | angle between the two links on the front right               | ankle_2 (front_right_leg)              | hinge |
    | 11  | angle between torso and first link on back left              | hip_3 (back_leg)                       | hinge |
    | 12  | angle between the two links on the back left                 | ankle_3 (back_leg)                     | hinge |
    | 13  | angle between torso and first link on back right             | hip_4 (right_back_leg)                 | hinge |
    | 14  | angle between the two links on the back right                | ankle_4 (right_back_leg)               | hinge |
    | 15  | x-coordinate velocity of the torso                           | torso                                  | free  |
    | 16  | y-coordinate velocity of the torso                           | torso                                  | free  |
    | 17  | z-coordinate velocity of the torso                           | torso                                  | free  |
    | 18  | x-coordinate angular velocity of the torso                   | torso                                  | free  |
    | 19  | y-coordinate angular velocity of the torso                   | torso                                  | free  |
    | 20  | z-coordinate angular velocity of the torso                   | torso                                  | free  |
    | 21  | angular velocity of angle between torso and front left link  | hip_1 (front_left_leg)                 | hinge |
    | 22  | angular velocity of the angle between front left links       | ankle_1 (front_left_leg)               | hinge |
    | 23  | angular velocity of angle between torso and front right link | hip_2 (front_right_leg)                | hinge |
    | 24  | angular velocity of the angle between front right links      | ankle_2 (front_right_leg)              | hinge |
    | 25  | angular velocity of angle between torso and back left link   | hip_3 (back_leg)                       | hinge |
    | 26  | angular velocity of the angle between back left links        | ankle_3 (back_leg)                     | hinge |
    | 27  | angular velocity of angle between torso and back right link  | hip_4 (right_back_leg)                 | hinge |
    | 28  |angular velocity of the angle between back right links        | ankle_4 (right_back_leg)               | hinge |
"""


"""
    | Num | Action                                                            | Name (in corresponding XML file) |
    | --- | ----------------------------------------------------------------- | -------------------------------- |
    | 0   | Torque applied on the rotor between the torso and front left hip  | hip_1 (front_left_leg)           |
    | 1   | Torque applied on the rotor between the front left two links      | ankle_1 (front_left_leg)         |
    | 2   | Torque applied on the rotor between the torso and front right hip | hip_2 (front_right_leg)          |
    | 3   | Torque applied on the rotor between the front right two links     | ankle_2 (front_right_leg)        |
    | 4   | Torque applied on the rotor between the torso and back left hip   | hip_3 (back_leg)                 |
    | 5   | Torque applied on the rotor between the back left two links       | ankle_3 (back_leg)               |
    | 6   | Torque applied on the rotor between the torso and back right hip  | hip_4 (right_back_leg)           |
    | 7   | Torque applied on the rotor between the back right two links      | ankle_4 (right_back_leg)         |
"""