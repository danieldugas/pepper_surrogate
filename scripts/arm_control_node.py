#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf.transformations as se3

from pepper_surrogate.msg import ButtonToggle
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import TransformStamped

import pepper_kinematics as pk

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
        # constants
        self.joint_names = pk.right_arm_tags if side == "right" else pk.left_arm_tags

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

    def update(self, se3_controller_in_vrroom):
        """
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
        # try to map human wrist yaw to pepper wrist yaw
        # get virtual claw (desired) in virtual wrist (after ik) frame
        if self.side == "right":
            arm_get_position = pk.right_arm_get_position
        else:
            arm_get_position = pk.left_arm_get_position
        # get position, orientation for every joint in arm
        pos_in_torso, ori_in_torso = arm_get_position(self.joint_angles, scale=self.scale, full_pos=True)
        idx = 4 # joint_frames = ["LShoulder", "LBicep", "LElbow", "LForeArm", "l_wrist", "LHand"]
        se3_virtual_wrist_in_virtual_torso = se3_from_pos_rot3(pos_in_torso[idx], ori_in_torso[idx])
        se3_virtual_torso_in_virtual_wrist = se3.inverse_matrix(se3_virtual_wrist_in_virtual_torso)
        se3_virtual_claw_in_virtual_wrist = np.dot(
            se3_virtual_torso_in_virtual_wrist,
            se3_virtual_claw_in_virtual_torso
        )
        # x is the distal direction in the wrist and virtual_claw tf
        wrist_yaw, _, _ = se3.euler_from_matrix(se3_virtual_claw_in_virtual_wrist)
        print(wrist_yaw)
        
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
        else:
            arm_get_position = pk.left_arm_get_position
            joint_frames = ["LShoulder", "LBicep", "LElbow", "LForeArm", "l_wrist", "LHand"]
            torso_frame = "virtualarm_LTorso"
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
    # state machine:
    # idle
    # moving arms to zero -> awaiting user ready -> initialize virtual arm
    # track virtual arm
    def __init__(self):
        rospy.init_node("stereo_cameras")

        # variables
        self.virtual_arms = {"right": None, "left": None}
        self.current_state = "idle"

        # publishers
        self.joint_angles_pub = rospy.Publisher("/pepper_robot/pose/joint_angles", JointAnglesWithSpeed, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # tf listener
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Subscribers
        rospy.Subscriber("/oculus/button_a_toggle", ButtonToggle, self.deadmanswitch_toggle_callback)
        rospy.Subscriber("/oculus/button_b_toggle", ButtonToggle, self.zeroswitch_toggle_callback)

        # Timer
        rospy.Timer(rospy.Duration(0.01), self.arm_update_routine)
        rospy.Timer(rospy.Duration(0.1), self.arm_control_routine)

    def is_zero_pose_reached(self):
        return True

    def visualize_vrhands(self):
        """ publishes the static transform from controller to vrhand """
        # Right
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kRightControllerFrame
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

    def arm_update_routine(self, event=None):
        # show vrhand
        self.visualize_vrhands()

        # for each hand
        for side in ["right", "left"]:
            controller_frame = kRightControllerFrame if side == "right" else kLeftControllerFrame
            # get controllers tf
            se3_controller_in_vrroom = None
            if self.current_state in ["tracking", "trackingactive", "zero"]:
                try:
                    trans = self.tf_buf.lookup_transform(kVRRoomFrame, controller_frame, rospy.Time())
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(e)
                    return
                se3_controller_in_vrroom = se3_from_transformstamped(trans)

            # update virtual arm angles
            if self.current_state in ["tracking", "trackingactive"]:
                # sometimes one arm did not get zeroed (controller not tracked). Tolerated for debug reasons
                if self.virtual_arms[side] is not None:
                    self.virtual_arms[side].update(se3_controller_in_vrroom)
                    self.virtual_arms[side].visualize(self.tf_br)

            if self.current_state == "zero":
                # create new temporary virtual arm and display it
                # when confirm button is pressed the temporary arm will become permanent and joints unlocked
                self.virtual_arms[side] = VirtualArm(side=side)
                self.virtual_arms[side].initialize_from_zero_pose_forward_kinematics(se3_controller_in_vrroom, self.tf_br)
                self.virtual_arms[side].visualize(self.tf_br)

    def arm_control_routine(self, event=None):
        # get both virtual arms joint angles, publish them
        if self.current_state == "trackingactive":
            joint_names = []
            joint_angles = []
            for side in ["right", "left"]:
                if self.virtual_arms[side] is not None:
                    joint_names.extend(self.virtual_arms[side].joint_names)
                    joint_angles.extend(self.virtual_arms[side].joint_angles)
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
            if self.current_state in ["idle", "tracking", "trackingactive"]:
                self.current_state = "zero"
            elif self.current_state == "zero":
                self.current_state = "idle"
            print("to ", self.current_state)
        else:
            pass

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


if __name__ == "__main__":
    node = ArmControlNode()
    rospy.spin()
