#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf.transformations as se3

from pepper_surrogate.msg import ButtonToggle, ACNDebugData
from std_msgs.msg import Float32
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import TransformStamped

import pepper_kinematics as pk
from vicon_arm_calibration import load_vicon_calibration

VIRTUALARM_SCALE = 1.8 # also change this value in urdf/01-virtualarm.xacro

# all angles in radians
left_arm_rest_pose = {
    'LShoulderPitch': 1.5,
    'LShoulderRoll': 0.,
    'LElbowYaw': 0.,
    'LElbowRoll': -0.,
    'LWristYaw': -1.,
}
right_arm_rest_pose = {
    'RShoulderPitch': 1.5,
    'RShoulderRoll': -0.,
    'RElbowYaw': 0.,
    'RElbowRoll': 0.,
    'RWristYaw': 1.,
}
left_arm_zero_pose = {
    'LShoulderPitch': 0.,
    'LShoulderRoll': 0.3,
    'LElbowYaw': 0.,
    'LElbowRoll': -0.3,
    'LWristYaw': 0.,
}
right_arm_zero_pose = {
    'RShoulderPitch': 0.,
    'RShoulderRoll': -0.3,
    'RElbowYaw': 0.,
    'RElbowRoll': 0.3,
    'RWristYaw': 0.,
}

kVRRoomFrame = "vrroom"
kRightControllerFrame = "oculus_right_controller"
kLeftControllerFrame = "oculus_left_controller"
kMaxArmSpeedRadPerSec = 0.1

kViconFrame = "vicon/world"
kViconTorsoFrame = None # we don't have a vicon marker there
kViconInferredTorsoFrame = "vicon/vicon_inferred_torso" # used only for publishing debug info
kRightViconArmbandFrame = "vicon/vicon_right_armband/vicon_right_armband"
kRightViconWristbandFrame = "vicon/vicon_right_wristband/vicon_right_wristband"
kLeftViconArmbandFrame = "vicon/vicon_left_armband/vicon_left_armband"
kLeftViconWristbandFrame = "vicon/vicon_left_wristband/vicon_left_wristband"
kRightViconControllerFrame = "vicon/oculus_right_controller"
kLeftViconControllerFrame = "vicon/oculus_left_controller"

DEBUG_TRANSFORMS = False # replaces true vicon transform with static made-up ones for testing
PUBLISH_DEBUG_TFS = True # publishes extra tfs which are useful for debugging
VRROOM_IN_VICON = se3.translation_matrix([1, 1, 0])

# controller frame != pepper hand frame even if the hands are superposed!
# find static transformation between controller frame and pepper hand if it was holding the controller
# rotate right along x axis (wrist axis)
se3_right_vrhand_in_controller = se3.rotation_matrix(np.pi / 2., np.array([1., 0., 0.]))
se3_right_vrhand_in_controller[2, 3] = -0.05
se3_left_vrhand_in_controller = se3.rotation_matrix(-np.pi / 2., np.array([1., 0., 0.]))
se3_left_vrhand_in_controller[2, 3] = -0.05

def se3_from_transformstamped(trans):
    """
    trans : TransformStamped
    M : 4x4 se3 matrix
    """
    transl = np.array([
        trans.transform.translation.x,
        trans.transform.translation.y,
        trans.transform.translation.z,
    ])
    quat = np.array([
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w,
    ])
    return se3_from_transl_quat(transl, quat)

def se3_from_transl_quat(transl, quat):
    """
    transl : [x, y, z]
    quat : [x, y, z, w]
    M : 4x4 se3 matrix
    """
    return np.dot(se3.translation_matrix(transl), se3.quaternion_matrix(quat))

def rot_mat_from_basis_vectors(x, y):
    """ right hand basis Z = X x Y
    x : [x y z]
    y : [x y z]
    M : 3x3 rot matrix"""
    xnorm = x / np.linalg.norm(x)
    ynorm = y / np.linalg.norm(y)
    # what if x and y are not perfectly orthogonal?
    if np.dot(xnorm, ynorm) > 0.01:
        print("Warning: basis vectors are not orthogonal")
    znorm = np.cross(xnorm, ynorm)
    rotmat = np.stack([xnorm, ynorm, znorm]).T
    return rotmat

def se3_from_pos_rot3(pos, rot):
    """
    pos : [x, y, z]
    rot : 3x3 rot matrix
    M : 4x4 se3 matrix
    """
    rot4 = se3.identity_matrix()
    rot4[:3, :3] = rot
    M = np.dot(se3.translation_matrix(pos), rot4)
    return M

class VirtualArm:
    """ A virtual arm created in the VRRoom to mirror pepper's actual arm.
    the virtual arm tracks the user's controller, and the resulting joint angles are sent
    to pepper control """
    def __init__(self, side="right"):
        # parameters
        self.side = side
        self.scale = VIRTUALARM_SCALE
        # variables
        self.se3_virtual_torso_in_vrroom = None
        self.joint_angles = None
        self.gripper_open = None
        self.debug_data = None
        # constants
        self.joint_names = pk.right_arm_tags if side == "right" else pk.left_arm_tags
        self.gripper_name = "RHand" if side == "right" else "LHand"

    def initialize_from_zero_pose_forward_kinematics(self, se3_controller_in_vrroom, tf_br):
        """ We know the angles for pepper's arms in zero pose.
        apply those from the controller position to get the torso position

        Notes:
        - TODO? use gravity to correct wrist rotation error
        - TODO? use hand-eye transform and claw-camera transform to infer scale
        """
        # left vs right functions and constants
        if self.side == "right":
            zero_pose_angles = right_arm_zero_pose.values()
            arm_get_position = pk.right_arm_get_position
            se3_vrhand_in_controller = se3_right_vrhand_in_controller
        else:
            zero_pose_angles = left_arm_zero_pose.values()
            arm_get_position = pk.left_arm_get_position
            se3_vrhand_in_controller = se3_left_vrhand_in_controller
        # initialize joint angles
        self.joint_angles = zero_pose_angles
        # forward kinematics
        se3_virtual_claw_in_virtual_torso = se3_from_pos_rot3(
            *arm_get_position(self.joint_angles, scale=self.scale))
        se3_virtual_torso_in_virtual_claw = se3.inverse_matrix(se3_virtual_claw_in_virtual_torso)
        # assume hand and claw are in the same place (user did a good job) to find virtual torso estimate
        se3_virtual_claw_in_vrhand = se3.identity_matrix()
        # compose tfs to get virtual torso in vrroom
        # vrroom -> controller -> vrhand -> virtual_claw -> virtual_torso
        se3_vrhand_in_vrroom = np.dot(
            se3_controller_in_vrroom,
            se3_vrhand_in_controller
        )
        se3_virtual_claw_in_vrroom = np.dot(
            se3_vrhand_in_vrroom,
            se3_virtual_claw_in_vrhand
        )
        se3_virtual_torso_in_vrroom = np.dot(
            se3_virtual_claw_in_vrroom,
            se3_virtual_torso_in_virtual_claw
        )
        self.se3_virtual_torso_in_vrroom = se3_virtual_torso_in_vrroom

    def vicon_update(self, se3_armband_in_vicon, se3_wristband_in_vicon, se3_torso_in_vicon,
                     vicon_calib, tf_br):
        """ Update virtual arm based on vicon tracking """
        v_T_ab = se3_armband_in_vicon
        v_T_wb = se3_wristband_in_vicon
        v_T_t = se3_torso_in_vicon
        if self.side == "right":
            t_T_s = vicon_calib.se3_right_shoulder_in_torso
            ab_T_e = vicon_calib.se3_right_elbow_in_right_armband
            wb_T_w = vicon_calib.se3_right_wrist_in_right_wristband
        else:
            t_T_s = vicon_calib.se3_left_shoulder_in_torso
            ab_T_e = vicon_calib.se3_left_elbow_in_left_armband
            wb_T_w = vicon_calib.se3_left_wrist_in_left_wristband
        # get shoulder, elbow, wrist transforms from calibration
        v_T_s = np.dot(v_T_t, t_T_s) # true orientation
        v_T_e = np.dot(v_T_ab, ab_T_e)
        v_T_w = np.dot(v_T_wb, wb_T_w)
        # position-only in vicon frame
        s = se3.translation_from_matrix(v_T_s)
        e = se3.translation_from_matrix(v_T_e)
        w = se3.translation_from_matrix(v_T_w)
        # calculate elbow roll
        ew = w - e
        se = e - s
        elbow_roll = np.arccos(np.dot(ew, se) / (np.linalg.norm(ew) * np.linalg.norm(se)))
        if self.side == "left":
            elbow_roll = -elbow_roll
        # calculate shoulder roll
        s_T_v = se3.inverse_matrix(v_T_s)
        s_T_e = np.dot(s_T_v, v_T_e)
        x, y, z = se3.translation_from_matrix(s_T_e) # in shoulder frame
        if self.side == "right":
            shoulder_roll = np.arccos(x / np.sqrt(x*x + y*y + z*z)) - np.pi / 2.
        else:
            shoulder_roll = np.pi / 2. - np.arccos(-x / np.sqrt(x*x + y*y + z*z))
        # calculate shoulder pitch
        shoulder_pitch = 0
        shoulder_singularity = np.abs(shoulder_roll) >= np.deg2rad(85)
        if not shoulder_singularity:
            shoulder_pitch = -np.arctan2(z, y)
        else: # singularity: elbow aligned with soulder, any shoulder pitch is possible
            if self.joint_angles is not None:
                SHLDR_PTCH_INDX = 0
                shoulder_pitch = self.joint_angles[SHLDR_PTCH_INDX]
            # apply angle limits since singularity allows any value
            if self.side == "right":
                shoulder_pitch_limits = [pk.get_right_arm_min_angles()[SHLDR_PTCH_INDX],
                                         pk.get_right_arm_max_angles()[SHLDR_PTCH_INDX]]
            else:
                shoulder_pitch_limits = [pk.get_left_arm_min_angles()[SHLDR_PTCH_INDX],
                                         pk.get_left_arm_max_angles()[SHLDR_PTCH_INDX]]
            shoulder_pitch = np.clip(shoulder_pitch, shoulder_pitch_limits[0], shoulder_pitch_limits[1])
        # forward kinematics to get pepper arm orientation from shoulder
        def fk_arm_in_shoulder(shoulder_pitch, shoulder_roll, side):
            """ sb: shoulder ball, ao: arm oriented, a: arm """
            x_angle = -shoulder_pitch+np.pi/2.
            s_T_sb = se3_from_pos_rot3(np.array([0,0,0]), se3.euler_matrix(x_angle, 0, 0)[:3, :3])
            if side == "right":
                y_angle = np.pi/2.+shoulder_roll
                sb_T_ao = se3_from_pos_rot3(np.array([0,0,0]), se3.euler_matrix(0, y_angle, 0)[:3, :3])
                ao_T_a = se3_from_pos_rot3(np.array([0.1,0,0]), se3.euler_matrix(0, 0, 0)[:3, :3])
            else:
                y_angle = -np.pi/2.+shoulder_roll
                sb_T_ao = se3_from_pos_rot3(np.array([0,0,0]), se3.euler_matrix(0, y_angle, 0)[:3, :3])
                ao_T_a = se3_from_pos_rot3(np.array([-0.1,0,0]), se3.euler_matrix(0, 0, 0)[:3, :3])
            s_T_a = np.dot(np.dot(s_T_sb, sb_T_ao), ao_T_a)
            return s_T_a, s_T_sb
        s_T_a, s_T_sb = fk_arm_in_shoulder(shoulder_pitch, shoulder_roll, self.side)
        v_T_a = np.dot(v_T_s, s_T_a)
        a_T_v = se3.inverse_matrix(v_T_a)
        # calculate elbow yaw
        if not vicon_calib.is_armband_orientation_calibrated:
            a_T_w = np.dot(a_T_v, v_T_w)
            a_T_e = np.dot(a_T_v, v_T_e)
            a_w = se3.translation_from_matrix(a_T_w) # in pepper arm frame
            a_e = se3.translation_from_matrix(a_T_e)
            x, y, z = a_w - a_e
            elbow_yaw = 0
            elbow_singularity = np.abs(elbow_roll) <= np.deg2rad(5)
            # detect singularity around shoulder axis
            if not elbow_singularity:
                if self.side == "right":
                    elbow_yaw = np.arctan2(z, y)
                else:
                    elbow_yaw = np.arctan2(-z, y)
            else: # wrist aligned with elbow
                if self.joint_angles is not None:
                    ELBW_YAW_INDX = 2
                    elbow_yaw = self.joint_angles[ELBW_YAW_INDX]
                # if the shoulder is singular, and elbow exceeds limits,
                # spin shoulder until elbow is within limits again
                if shoulder_singularity:
                    ELBW_YAW_INDX = 2
                    if self.side == "right":
                        elbow_yaw_limits = [pk.get_right_arm_min_angles()[ELBW_YAW_INDX],
                                            pk.get_right_arm_max_angles()[ELBW_YAW_INDX]]
                    else:
                        elbow_yaw_limits = [pk.get_left_arm_min_angles()[ELBW_YAW_INDX],
                                            pk.get_left_arm_max_angles()[ELBW_YAW_INDX]]
                    new_elbow_yaw = np.clip(elbow_yaw, elbow_yaw_limits[0], elbow_yaw_limits[1])
                    elbow_yaw_correction = new_elbow_yaw - elbow_yaw
                    if self.side == "right": # on the right side, shoulder pitch and elbow_yaw are opposed
                        shoulder_pitch += elbow_yaw_correction
                    else:
                        shoulder_pitch -= elbow_yaw_correction
                    elbow_yaw = new_elbow_yaw
                    # recalculate arm with modified shoulder pitch
                    s_T_a, s_T_sb = fk_arm_in_shoulder(shoulder_pitch, shoulder_roll, self.side)
                    v_T_a = np.dot(v_T_s, s_T_a)
        else: # elbow yaw from calibrated armband orientation (v_T_ab points toward elbow rotation plane)
            a_T_ab = np.dot(a_T_v, v_T_ab)
            elbow_yaw = se3.euler_from_matrix(a_T_ab)[0]
        # try to map human wrist yaw to pepper wrist yaw
        # get armband (desired) in virtual wrist (after ik) frame
        if self.side == "right":
            arm_get_position = pk.right_arm_get_position
        else:
            arm_get_position = pk.left_arm_get_position
        self.joint_angles = [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll, 0]
        # get wrist yaw (forward kinematics to wrist)
        # x is the distal direction in the wrist and virtual_claw tf
        pos_in_torso, ori_in_torso = arm_get_position(self.joint_angles, scale=self.scale, full_pos=True)
        FRARM_IDX = 3 # joint_frames = ["LShoulder", "LBicep", "LElbow", "LForeArm", "l_wrist", "LHand"]
        se3_virtual_forearm_in_virtual_torso = se3_from_pos_rot3(pos_in_torso[FRARM_IDX],
                                                                 ori_in_torso[FRARM_IDX])
        se3_virtual_torso_in_virtual_forearm = se3.inverse_matrix(se3_virtual_forearm_in_virtual_torso)
        # update shoulder position virtual_torso-> virtual_shoulder == shoulder
        # vicon shoulder is fixed wrt torso, but virtual shoulder is not (same as armball)
        # therefore shoulder <-static-> virtual_torso -> virtual shoulder
        # also vrroom <-> vicon
        d = pk.fk.joint_distances[0] * self.scale
        if self.side == "right":
            s_T_vt = se3_from_pos_rot3(np.array([-d, 0, 0]), se3.euler_matrix(0, 0, np.pi/2.)[:3, :3])
        if self.side == "left":
            s_T_vt = se3_from_pos_rot3(np.array([d, 0, 0]), se3.euler_matrix(0, 0, np.pi/2.)[:3, :3])
        v_T_vt = np.dot(v_T_s, s_T_vt) # virtual torso in vicon
        vr_T_v = se3.inverse_matrix(VRROOM_IN_VICON)
        vr_T_vt = np.dot(vr_T_v, v_T_vt)
        self.se3_virtual_torso_in_vrroom = vr_T_vt
        vt_T_v = se3.inverse_matrix(v_T_vt)
        # we need to transform from wristband -> controller -> vrhand
        if self.side == "right":
            wb_T_c = se3.identity_matrix()
            c_frame = kRightViconControllerFrame
            c_T_h = se3_right_vrhand_in_controller
        else:
            wb_T_c = se3.identity_matrix()
            c_frame = kLeftViconControllerFrame
            c_T_h = se3_left_vrhand_in_controller
        v_T_c = np.dot(v_T_wb, wb_T_c)
        v_T_h = np.dot(v_T_c, c_T_h)
        vt_T_h = np.dot(vt_T_v, v_T_h)
        se3_vrhand_in_virtual_torso = vt_T_h
        se3_vrhand_in_virtual_forearm = np.dot(
            se3_virtual_torso_in_virtual_forearm,
            se3_vrhand_in_virtual_torso
        )
        wrist_yaw, _, _ = se3.euler_from_matrix(se3_vrhand_in_virtual_forearm)
        # if the elbow is singular, and wrist exceeds limits, spin elbow until wrist is within limits again
        if elbow_singularity:
            WRST_ANG_IDX = 4
            if self.side == "right":
                yaw_limits = [pk.get_right_arm_min_angles()[WRST_ANG_IDX],
                              pk.get_right_arm_max_angles()[WRST_ANG_IDX]]
            else:
                arm_get_position = pk.left_arm_get_position
                yaw_limits = [pk.get_left_arm_min_angles()[WRST_ANG_IDX],
                              pk.get_left_arm_max_angles()[WRST_ANG_IDX]]
            new_wrist_yaw = np.clip(wrist_yaw, yaw_limits[0], yaw_limits[1])
            wrist_yaw_correction = new_wrist_yaw - wrist_yaw
            elbow_yaw -= wrist_yaw_correction
            wrist_yaw = new_wrist_yaw
        self.joint_angles = [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll, wrist_yaw]
        # publish debug data
        if self.side == "right":
            joint_limits = np.array([pk.get_right_arm_min_angles(self.joint_angles),
                                     pk.get_right_arm_max_angles(self.joint_angles)])
        else:
            joint_limits = np.array([pk.get_left_arm_min_angles(self.joint_angles),
                                     pk.get_left_arm_max_angles(self.joint_angles)])
        self.debug_data = {
            "time":                  rospy.Time.now(),
            "shoulder_pitch":        shoulder_pitch,
            "shoulder_roll":         shoulder_roll,
            "elbow_yaw":             elbow_yaw,
            "elbow_roll":            elbow_roll,
            "wrist_yaw":             wrist_yaw,
            "min_shoulder_pitch":    joint_limits[0, 0],
            "min_shoulder_roll":     joint_limits[0, 1],
            "min_elbow_yaw":         joint_limits[0, 2],
            "min_elbow_roll":        joint_limits[0, 3],
            "min_wrist_yaw":         joint_limits[0, 4],
            "max_shoulder_pitch":    joint_limits[1, 0],
            "max_shoulder_roll":     joint_limits[1, 1],
            "max_elbow_yaw":         joint_limits[1, 2],
            "max_elbow_roll":        joint_limits[1, 3],
            "max_wrist_yaw":         joint_limits[1, 4],
            "shoulder_singularity":  float(shoulder_singularity),
            "elbow_singularity":     float(elbow_singularity),
            "torso_in_vicon":        se3.translation_from_matrix(v_T_vt),
        }
        # publish tfs for joints used in vicon calculations
        # at the end, we publish a static tf for vrroom in vicon, which allows showing the virtual arms
        v_T_vr = VRROOM_IN_VICON
        vr_T_v = se3.inverse_matrix(v_T_vr)
        T = vr_T_v
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kVRRoomFrame
        t.child_frame_id = kViconFrame
        pos = se3.translation_from_matrix(T)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(T)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        tf_br.sendTransform(t)
        # publish if desired
        if PUBLISH_DEBUG_TFS:
            for T, name in zip(
                [v_T_s, v_T_e, v_T_w, np.dot(v_T_s, s_T_sb), v_T_a, v_T_c],
                ["vicon_s", "vicon_e", "vicon_w", "vicon_sb", "vicon_a", c_frame],
            ):
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = kViconFrame
                t.child_frame_id = self.side+name if name != c_frame else name
                pos = se3.translation_from_matrix(T)
                t.transform.translation.x = pos[0]
                t.transform.translation.y = pos[1]
                t.transform.translation.z = pos[2]
                q = se3.quaternion_from_matrix(T)
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]
                tf_br.sendTransform(t)

    def update(self, se3_controller_in_vrroom):
        """
        Update virtual arm based on oculus controller position

        controller has moved in virtual torso frame, move the virtual hand the same amount and get
        new joint angles

        Notes:
        - use headset movement to infer torso drift
        """
        # left vs right functions and constants
        if self.side == "right":
#             arm_set_position = pk.right_arm_set_position
            arm_ik_single_iteration = pk.right_arm_ik_single_iteration
            se3_vrhand_in_controller = se3_right_vrhand_in_controller
        else:
#             arm_set_position = pk.left_arm_set_position
            arm_ik_single_iteration = pk.left_arm_ik_single_iteration
            se3_vrhand_in_controller = se3_left_vrhand_in_controller
        # compose tfs to get virtual_claw in virtual_torso
        # vrroom -> controller -> vrhand -> virtual_claw
        # vrroom -> virtual_torso
        se3_vrhand_in_vrroom = np.dot(
            se3_controller_in_vrroom,
            se3_vrhand_in_controller
        )
        se3_virtual_claw_in_vrhand = se3.identity_matrix() # user hand and robot hand are now glued
        se3_virtual_claw_in_vrroom = np.dot(
            se3_vrhand_in_vrroom,
            se3_virtual_claw_in_vrhand
        )
        se3_vrroom_in_virtual_torso = se3.inverse_matrix(self.se3_virtual_torso_in_vrroom)
        se3_virtual_claw_in_virtual_torso = np.dot(
            se3_vrroom_in_virtual_torso,
            se3_virtual_claw_in_vrroom,
        )
        new_pos = se3.translation_from_matrix(se3_virtual_claw_in_virtual_torso)
        new_rot = se3_virtual_claw_in_virtual_torso[:3, :3]
#         new_angles = arm_set_position(self.joint_angles, new_pos, new_rot, epsilon = 0.1)
        # inverse kinematics (using jacobian) - limited to 3DOF end-effector (no rotation)
        # human hand is much more flexible than pepper's. Mapping hand rotations to joint angles
        # is not straightforward.
        new_angles = arm_ik_single_iteration(self.joint_angles, new_pos, new_rot, scale=self.scale)
        if new_angles is None:
            return
        # -------------
        # try to map human wrist yaw to pepper wrist yaw
        # get virtual claw (desired) in virtual wrist (after ik) frame
        WRST_ANG_IDX = 4
        if self.side == "right":
            arm_get_position = pk.right_arm_get_position
            yaw_limits = [pk.get_right_arm_min_angles()[WRST_ANG_IDX],
                          pk.get_right_arm_max_angles()[WRST_ANG_IDX]]
        else:
            arm_get_position = pk.left_arm_get_position
            yaw_limits = [pk.get_left_arm_min_angles()[WRST_ANG_IDX],
                          pk.get_left_arm_max_angles()[WRST_ANG_IDX]]
        # get position, orientation for every joint in arm
        pos_in_torso, ori_in_torso = arm_get_position(self.joint_angles, scale=self.scale, full_pos=True)
        FRARM_IDX = 3 # joint_frames = ["LShoulder", "LBicep", "LElbow", "LForeArm", "l_wrist", "LHand"]
        se3_virtual_forearm_in_virtual_torso = se3_from_pos_rot3(pos_in_torso[FRARM_IDX],
                                                                 ori_in_torso[FRARM_IDX])
        se3_virtual_torso_in_virtual_forearm = se3.inverse_matrix(se3_virtual_forearm_in_virtual_torso)
        se3_virtual_claw_in_virtual_forearm = np.dot(
            se3_virtual_torso_in_virtual_forearm,
            se3_virtual_claw_in_virtual_torso
        )
        # x is the distal direction in the wrist and virtual_claw tf
        wrist_yaw, _, _ = se3.euler_from_matrix(se3_virtual_claw_in_virtual_forearm)
        wrist_yaw = np.clip(wrist_yaw, yaw_limits[0], yaw_limits[1])
        new_angles[WRST_ANG_IDX] = wrist_yaw
        # actualize angles
        self.joint_angles = new_angles

    def visualize(self, tf_br):
        """ show the virtual arm in rviz
        """
        # left vs right functions and constants
        if self.side == "right":
            arm_get_position = pk.right_arm_get_position
            joint_frames = ["RShoulder", "RBicep", "RElbow", "RForeArm", "r_wrist", "RHand"]
            torso_frame = "virtualarm_RTorso"
            gripper_frame = "virtualarm_r_gripper"
        else:
            arm_get_position = pk.left_arm_get_position
            joint_frames = ["LShoulder", "LBicep", "LElbow", "LForeArm", "l_wrist", "LHand"]
            torso_frame = "virtualarm_LTorso"
            gripper_frame = "virtualarm_l_gripper"
        # get position, orientation for every joint in arm
        pos_in_torso, ori_in_torso = arm_get_position(self.joint_angles, scale=self.scale, full_pos=True)
        time = rospy.Time.now()
        # publish tfs
        for pos, rot, frame in zip(pos_in_torso, ori_in_torso, joint_frames):
            t = TransformStamped()
            t.header.stamp = time
            t.header.frame_id = torso_frame
            t.child_frame_id = "virtualarm_" + frame
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            q = se3.quaternion_from_matrix(se3_from_pos_rot3(pos, rot))
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            tf_br.sendTransform(t)
        # publish fingers tf
        if self.gripper_open is not None:
            t = TransformStamped()
            t.header.stamp = time
            t.header.frame_id = "virtualarm_" + joint_frames[-1]
            t.child_frame_id = gripper_frame
            t.transform.translation.x = 0.04
            t.transform.translation.y = 0.
            t.transform.translation.z = 0.05
            q = se3.quaternion_from_euler(0, -self.gripper_open * np.pi / 2., 0)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            tf_br.sendTransform(t)
        # publish torso in vrroom
        t = TransformStamped()
        t.header.stamp = time
        t.header.frame_id = kVRRoomFrame
        t.child_frame_id = torso_frame
        pos = se3.translation_from_matrix(self.se3_virtual_torso_in_vrroom)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(self.se3_virtual_torso_in_vrroom)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        tf_br.sendTransform(t)


class ArmControlNode:
    """
    state machine:
        idle - arms down
        zero (oculus tracking only) - arms up, continuously initializes virtual arm
        tracking - arms at latest position
        tracking active - arms follow user movements

    oculus:
        idle -> zero -> tracking active <-> tracking
          ^---------------------'---------------'

    vicon:
        idle -> tracking active <-> tracking
          ^----------------------------'

    moving arms to zero -> awaiting user ready -> initialize virtual arm
    """
    def __init__(self):
        rospy.init_node("arm_control_node")

        # options
        self.vicon_tracking = rospy.get_param("~vicon_tracking", default=False)
        self.vicon_calib_file = "config/vicon_calibration.pckl"
        self.vicon_calib = load_vicon_calibration(self.vicon_calib_file) if self.vicon_tracking else None

        # variables
        self.virtual_arms = {"right": None, "left": None}
        self.current_state = "idle"

        # publishers
        self.joint_angles_pub = rospy.Publisher("/pepper_robot/pose/joint_angles",
                                                JointAnglesWithSpeed, queue_size=1)
        self.debug_data_pub = rospy.Publisher("/arm_control_node/debug_data",
                                              ACNDebugData, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # tf listener
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Subscribers
        rospy.Subscriber("/oculus/button_a_toggle", ButtonToggle, self.deadmanswitch_toggle_callback)
        rospy.Subscriber("/oculus/button_b_toggle", ButtonToggle, self.zeroswitch_toggle_callback)
        rospy.Subscriber("/oculus/left_gripper", Float32, self.left_gripperswitch_toggle_callback)
        rospy.Subscriber("/oculus/right_gripper", Float32, self.right_gripperswitch_toggle_callback)

        # Timer
        rospy.Timer(rospy.Duration(0.01), self.arm_update_routine)
        rospy.Timer(rospy.Duration(0.1), self.arm_control_routine)

        # Shutdown hook
        rospy.on_shutdown(self.on_exit_arm_reset)

    def is_zero_pose_reached(self):
        return True

    def visualize_vrhands(self):
        """ publishes the static transform from controller to vrhand """
        # Right
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kRightControllerFrame
        if self.vicon_tracking:
            t.header.frame_id = kRightViconControllerFrame
        t.child_frame_id = "oculus_right_vrhand"
        pos = se3.translation_from_matrix(se3_right_vrhand_in_controller)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(se3_right_vrhand_in_controller)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)
        # Left
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kLeftControllerFrame
        if self.vicon_tracking:
            t.header.frame_id = kLeftViconControllerFrame
        t.child_frame_id = "oculus_left_vrhand"
        pos = se3.translation_from_matrix(se3_left_vrhand_in_controller)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(se3_left_vrhand_in_controller)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)

    def get_latest_tf(self, parent_frame, child_frame):
        if child_frame is None:
            return None
        se3_child_in_parent = None
        if DEBUG_TRANSFORMS:
            if child_frame == kRightViconArmbandFrame:
                return se3_from_pos_rot3(np.array([1, 0.2, 0]), se3.euler_matrix(0,0.9,0.8)[:3, :3])
            if child_frame == kRightViconWristbandFrame:
                return se3_from_pos_rot3(np.array([1.5, 1., 0]), se3.euler_matrix(0,0,0.3)[:3, :3])
            if child_frame == kLeftViconArmbandFrame:
                return se3_from_pos_rot3(np.array([0, 0, 0]), se3.euler_matrix(0,0.9,np.pi)[:3, :3])
            if child_frame == kLeftViconWristbandFrame:
                return se3_from_pos_rot3(np.array([-0.5, 0.3, 0]), se3.euler_matrix(0,0.9,np.pi)[:3, :3])
        try:
            trans = self.tf_buf.lookup_transform(parent_frame, child_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print(e)
            return None
        se3_child_in_parent = se3_from_transformstamped(trans)
        return se3_child_in_parent

    def infer_torso_frame(self):
        """ If we don't have a torso vicon marker, we need to guess where the torso is pointing based on
        the two shoulder positions, and gravity / head placement """
        se3_left_armband_in_vicon = self.get_latest_tf(kViconFrame, kLeftViconArmbandFrame)
        se3_right_armband_in_vicon = self.get_latest_tf(kViconFrame, kRightViconArmbandFrame)
        # get shoulder positions vicon->armband->shoulder - FO means false orientation
        se3_left_shoulder_FO_in_vicon = np.dot(
            se3_left_armband_in_vicon, self.vicon_calib.se3_left_shoulder_in_left_armband)
        se3_right_shoulder_FO_in_vicon = np.dot(
            se3_right_armband_in_vicon, self.vicon_calib.se3_right_shoulder_in_right_armband)
        # calculate inferred torso
        sl = se3.translation_from_matrix(se3_left_shoulder_FO_in_vicon)
        sr = se3.translation_from_matrix(se3_right_shoulder_FO_in_vicon)
        t = (sl + sr) / 2.
        right = sr - sl # points right w.r.t human frame (lateral)
        up = np.array([0, 0, 1]) # up w.r.t human frame (but gravity aligned since we don't know torso pitch)
        forward = -np.cross(right, up) # points forward w.r.t human frame (anterior)
        t_rot = rot_mat_from_basis_vectors(right, forward)
        se3_torso_in_vicon = se3_from_pos_rot3(t, t_rot)
        # update torso -> shoulder calib translation # tTs = tTv vTs
        se3_right_shoulder_FO_in_torso = np.dot(
            se3.inverse_matrix(se3_torso_in_vicon),
            se3_right_shoulder_FO_in_vicon)
        se3_left_shoulder_FO_in_torso = np.dot(
            se3.inverse_matrix(se3_torso_in_vicon),
            se3_left_shoulder_FO_in_vicon)
        ls_in_torso = se3.translation_from_matrix(se3_left_shoulder_FO_in_torso)
        rs_in_torso = se3.translation_from_matrix(se3_right_shoulder_FO_in_torso)
        rs_rot_in_torso = np.identity(3) # replace false shoulder orientation with true one
        ls_rot_in_torso = np.identity(3)  # replace false shoulder orientation with true one
        se3_left_shoulder_in_torso = se3_from_pos_rot3(ls_in_torso, ls_rot_in_torso)
        se3_right_shoulder_in_torso = se3_from_pos_rot3(rs_in_torso, rs_rot_in_torso)
        self.vicon_calib.se3_left_shoulder_in_torso = se3_left_shoulder_in_torso
        self.vicon_calib.se3_right_shoulder_in_torso = se3_right_shoulder_in_torso
        # publish inferred torso tf for debugging
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kViconFrame
        t.child_frame_id = kViconInferredTorsoFrame
        pos = se3.translation_from_matrix(se3_torso_in_vicon)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(se3_torso_in_vicon)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)
        return se3_torso_in_vicon

    def arm_update_routine(self, event=None):
        # show vrhand
        self.visualize_vrhands()

        # for each hand
        for side in ["right", "left"]:
            if self.vicon_tracking:
                # get tfs of tracking markers (oculus controllers or vicon)
                if self.virtual_arms[side] is None:
                    self.virtual_arms[side] = VirtualArm(side=side)
                armband_frame = kRightViconArmbandFrame if side == "right" else kLeftViconArmbandFrame
                wristband_frame = (kRightViconWristbandFrame if side == "right"
                                   else kLeftViconWristbandFrame)
                torso_frame = kViconTorsoFrame
                se3_armband_in_vicon = self.get_latest_tf(kViconFrame, armband_frame)
                se3_wristband_in_vicon = self.get_latest_tf(kViconFrame, wristband_frame)
                se3_torso_in_vicon = self.get_latest_tf(kViconFrame, torso_frame)
                if se3_armband_in_vicon is None or se3_wristband_in_vicon is None:
                    rospy.logwarn_throttle(2, side + " arm vicon marker transforms not found")
                    continue
                if se3_torso_in_vicon is None or self.vicon_calib.se3_left_shoulder_in_torso is None:
                    se3_torso_in_vicon = self.infer_torso_frame()

                # update virtual arm angles
                self.virtual_arms[side].vicon_update(se3_armband_in_vicon, se3_wristband_in_vicon,
                                                     se3_torso_in_vicon, self.vicon_calib,
                                                     self.tf_br
                                                     )
                self.virtual_arms[side].visualize(self.tf_br)
            else:
                # get tfs of tracking markers (oculus controllers or vicon)
                controller_frame = kRightControllerFrame if side == "right" else kLeftControllerFrame
                se3_controller_in_vrroom = self.get_latest_tf(kVRRoomFrame, controller_frame)
                if se3_controller_in_vrroom is None:
                    rospy.logwarn_throttle(2, side + " controller transforms not found")
                    continue

                # update virtual arm angles
                if self.current_state in ["tracking", "trackingactive"]:
                    # sometimes one arm did not get zeroed (controller not tracked).
                    # Tolerated for debug reasons
                    if self.virtual_arms[side] is None:
                        rospy.logwarn_throttle(2, side + " arm is not initialized. \
                                               Please switch to zero pose.")
                    else:
                        self.virtual_arms[side].update(se3_controller_in_vrroom)
                        self.virtual_arms[side].visualize(self.tf_br)

                if self.current_state == "zero":
                    # create new temporary virtual arm and display it
                    # when confirm button is pressed,
                    # the temporary arm will become permanent and joints unlocked
                    self.virtual_arms[side] = VirtualArm(side=side)
                    self.virtual_arms[side].initialize_from_zero_pose_forward_kinematics(
                        se3_controller_in_vrroom, self.tf_br)
                    self.virtual_arms[side].visualize(self.tf_br)

        self.publish_debug_data()

    def publish_debug_data(self):
        if self.vicon_tracking:
            debug_data_msg = ACNDebugData()
            debug_data_msg.header.stamp = rospy.Time.now()
            keys = []
            values = []
            torsos = {"right": None, "left": None}
            for side in ["right", "left"]:
                if self.virtual_arms[side] is None:
                    continue
                D = self.virtual_arms[side].debug_data
                if D is None:
                    continue
                for key in D:
                    value = D[key]
                    if isinstance(value, float):
                        keys.append("{}_{}".format(side, key))
                        values.append(value)
                torsos[side] = D["torso_in_vicon"]
            if torsos["right"] is not None and torsos["left"] is not None:
                torso_dist = np.linalg.norm(torsos["right"] - torsos["left"])
                keys.append("torsos_dist")
                values.append(torso_dist)
            for key, value in zip(keys, values):
                setattr(debug_data_msg, key, value)
            self.debug_data_pub.publish(debug_data_msg)

    def arm_control_routine(self, event=None):
        # get both virtual arms joint angles, publish them
        if self.current_state == "trackingactive":
            joint_names = []
            joint_angles = []
            for side in ["right", "left"]:
                if self.virtual_arms[side] is not None:
                    joint_names.extend(self.virtual_arms[side].joint_names)
                    joint_angles.extend(self.virtual_arms[side].joint_angles)
                    if self.virtual_arms[side].gripper_open is not None:
                        joint_names.append(self.virtual_arms[side].gripper_name)
                        joint_angles.append(self.virtual_arms[side].gripper_open)
            if joint_angles:
                joint_angles_msg = JointAnglesWithSpeed()
                joint_angles_msg.speed = kMaxArmSpeedRadPerSec
                joint_angles_msg.joint_names = joint_names
                joint_angles_msg.joint_angles = joint_angles
                self.joint_angles_pub.publish(joint_angles_msg)
        if self.current_state == "zero":
            joint_names = []
            joint_angles = []
            joint_names.extend(left_arm_zero_pose.keys())
            joint_angles.extend(left_arm_zero_pose.values())
            joint_names.extend(right_arm_zero_pose.keys())
            joint_angles.extend(right_arm_zero_pose.values())
            joint_angles_msg = JointAnglesWithSpeed()
            joint_angles_msg.speed = kMaxArmSpeedRadPerSec
            joint_angles_msg.joint_names = joint_names
            joint_angles_msg.joint_angles = joint_angles
            self.joint_angles_pub.publish(joint_angles_msg)
        if self.current_state == "idle":
            joint_names = []
            joint_angles = []
            joint_names.extend(left_arm_rest_pose.keys())
            joint_angles.extend(left_arm_rest_pose.values())
            joint_names.extend(right_arm_rest_pose.keys())
            joint_angles.extend(right_arm_rest_pose.values())
            joint_angles_msg = JointAnglesWithSpeed()
            joint_angles_msg.speed = kMaxArmSpeedRadPerSec
            joint_angles_msg.joint_names = joint_names
            joint_angles_msg.joint_angles = joint_angles
            self.joint_angles_pub.publish(joint_angles_msg)

    def zeroswitch_toggle_callback(self, msg):
        if msg.event == ButtonToggle.PRESSED:
            print("Switching from ", self.current_state)
            if self.current_state in ["zero", "tracking", "trackingactive"]:
                self.current_state = "idle"
            elif self.current_state == "idle":
                self.current_state = "zero"
            print("to ", self.current_state)
        else:
            pass

    def right_gripperswitch_toggle_callback(self, msg):
        if self.virtual_arms["right"] is not None:
            self.virtual_arms["right"].gripper_open = np.clip(1.-msg.data, 0, 1)

    def left_gripperswitch_toggle_callback(self, msg):
        if self.virtual_arms["left"] is not None:
            self.virtual_arms["left"].gripper_open = np.clip(1.-msg.data, 0, 1)

    def deadmanswitch_toggle_callback(self, msg):
        if msg.event == ButtonToggle.PRESSED:
            if self.current_state == "zero":
                # check if zero pose is reached
                if self.is_zero_pose_reached():
                    print("Switching from ", self.current_state)
                    self.current_state = "trackingactive"
                    print("to ", self.current_state)
            elif self.current_state == "tracking":
                print("Switching from ", self.current_state)
                self.current_state = "trackingactive"
                print("to ", self.current_state)
            else:
                pass
        elif msg.event == ButtonToggle.RELEASED:
            if self.current_state == "trackingactive":
                print("Switching from ", self.current_state)
                self.current_state = "tracking"
                print("to ", self.current_state)

    def on_exit_arm_reset(self):
        self.current_state = "idle"
        joint_names = []
        joint_angles = []
        joint_names.extend(left_arm_rest_pose.keys())
        joint_angles.extend(left_arm_rest_pose.values())
        joint_names.extend(right_arm_rest_pose.keys())
        joint_angles.extend(right_arm_rest_pose.values())
        joint_angles_msg = JointAnglesWithSpeed()
        joint_angles_msg.speed = kMaxArmSpeedRadPerSec
        joint_angles_msg.joint_names = joint_names
        joint_angles_msg.joint_angles = joint_angles
        rospy.loginfo("Publishing rest pose before exiting.")
        self.joint_angles_pub.publish(joint_angles_msg)
        rospy.sleep(0.1)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    node = ArmControlNode()
    rospy.spin()
