import numpy as np

right_arm_tags = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
right_arm_initial_pose = [1.0, -0.2, 1.57-0.2, 1.0, -1.57]
right_arm_work_pose = [0.8, -0.2, 1.57-0.2, 0.9, -1.57]

_inverse_case = [1.0, -1.0, -1.0, -1.0, -1.0]

left_arm_tags = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
left_arm_initial_pose = [p[0] * p[1] for p in zip(right_arm_initial_pose, _inverse_case)]
left_arm_work_pose = [p[0] * p[1] for p in zip(right_arm_work_pose, _inverse_case)]

def get_left_arm_min_angles(angles=None):
    """ arm angle limits depending on current angle """
    static_limits = [-2.0857, 0.0087, -2.0857, -1.5620, -1.8239]
    if angles is None:
        return static_limits
    else:
        static_limits[3] = LElbowRollMin(angles[2])
        return static_limits

def get_left_arm_max_angles(angles=None):
    """ arm angle limits depending on current angle """
    return [ 2.0857, 1.5620,  2.0857, -0.0087,  1.8239]

def get_right_arm_min_angles(angles=None):
    """ arm angle limits depending on current angle """
    return [-2.0857, -1.5620, -2.0857, 0.0087, -1.8239]

def get_right_arm_max_angles(angles=None):
    """ arm angle limits depending on current angle """
    static_limits = [ 2.0857,  0.0087,  2.0857, 1.5620,  1.8239]
    if angles is None:
        return static_limits
    else:
        static_limits[3] = RElbowRollMax(angles[2])
        return static_limits


def LElbowRollMin(LElbowYaw):
    elbowyaw_clamped_deg = np.rad2deg(np.clip(LElbowYaw,
                                              get_left_arm_min_angles()[2],
                                              get_left_arm_max_angles()[2]))
    if elbowyaw_clamped_deg < 0.:
        return np.deg2rad(-78.0)
    elif elbowyaw_clamped_deg > 99.5:
        return np.deg2rad(-83.0)
    else:
        return get_left_arm_min_angles()[3]

def RElbowRollMax(RElbowYaw):
    elbowyaw_clamped_deg = np.rad2deg(np.clip(RElbowYaw,
                                              get_right_arm_min_angles()[2],
                                              get_right_arm_max_angles()[2]))
    if elbowyaw_clamped_deg < -99.5:
        return np.deg2rad(83.0)
    elif elbowyaw_clamped_deg > 0:
        return np.deg2rad(78.0)
    else:
        return get_right_arm_max_angles()[3]

def clamp_joints(angles, right=True):
    if right:
        min_angles = get_right_arm_min_angles(angles)
        max_angles = get_right_arm_max_angles(angles)
    else:
        min_angles = get_left_arm_min_angles(angles)
        max_angles = get_left_arm_max_angles(angles)
    return np.clip(angles, min_angles, max_angles)
