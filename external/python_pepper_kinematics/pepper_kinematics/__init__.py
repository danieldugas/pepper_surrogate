import numpy as np
import scipy as sp
from scipy import linalg

import forward_kinematics as fk
import inverse_kinematics as ik
from joint_definitions import left_arm_tags, left_arm_initial_pose, right_arm_tags, right_arm_initial_pose


def right_arm_get_position(angles, full_pos=False, scale=1.):
    """
    Just calculate the position when joints on the pepper's right arm is in given positions

    Args:
      angles : Angles of right arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return fk.calc_fk_and_jacob(angles, jacob=False, right=True, scale=scale, full_pos=full_pos)

def left_arm_get_position(angles, full_pos=False, scale=1.):
    """
    Just calculate the position when joints on the pepper's left arm is in given positions

    Args:
      angles : Angles of left arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return fk.calc_fk_and_jacob(angles, jacob=False, right=False, scale=scale, full_pos=full_pos)

def right_arm_set_position(angles, target_pos, target_ori, epsilon=0.0001):
    """
    Just calculate the joint angles when the Pepper's right hand position is in the given position
    
    Args:
      angles : Use the initial position of calculation. Unit = radian
      target_pos : List. [Px, Py, Pz]. Unit is meter.
      target_ori : np.array([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]])
      epsilon    : The threshold. If the distance between calculation result and target_position is lower than epsilon, this returns value.
    
    Returns:
      A list of joint angles (Unit is radian). If calculation fails, return None.
    """
    return ik.calc_inv_pos(angles, target_pos, target_ori, epsilon, right=True)

def left_arm_set_position(angles, target_pos, target_ori, epsilon = 0.0001):
    return ik.calc_inv_pos(angles, target_pos, target_ori, epsilon, right=False)

def left_arm_ik_single_iteration(initial_angles, target_pos, target_ori, scale=1.):
    return ik.single_step_towards_target(initial_angles, target_pos, target_ori, scale=scale, right=False)

def right_arm_ik_single_iteration(initial_angles, target_pos, target_ori, scale=1.):
    return ik.single_step_towards_target(initial_angles, target_pos, target_ori, scale=scale, right=True)

