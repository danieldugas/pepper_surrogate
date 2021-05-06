#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf.transformations as se3

from pepper_surrogate.msg import ButtonToggle
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import TransformStamped

import pepper_kinematics as pk

# all angles in radians
left_arm_rest_pose = {tag: angle for (tag, angle) in zip(pk.left_arm_tags, pk.left_arm_initial_pose)}
right_arm_rest_pose = {tag: angle for (tag, angle) in zip(pk.right_arm_tags, pk.right_arm_initial_pose)}
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

# controller frame != pepper hand frame even if the hands are superposed!
# find static transformation between controller frame and pepper hand if it was holding the controller
se3_vrhand_in_controller = se3.rotation_matrix(np.pi / 2., np.array([1., 0., 0.]))

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
        self.scale = 1. # makes the virtual robot arm bigger to correspond with human size
        self.side = side
        # variables
        self.se3_virtual_torso_in_vrroom = None

    def initialize_from_zero_pose_forward_kinematics(self, se3_controller_in_vrroom, tf_br):
        """ We know the angles for pepper's arms in zero pose.
        apply those from the controller position to get the torso position

        Notes:
        - use gravity to correct wrist rotation error
        - use hand-eye transform and claw-camera transform to infer scale
        """
        self.joint_angles = right_arm_zero_pose.values()
        # forward kinematics
        se3_virtual_claw_in_virtual_torso = se3_from_pos_rot3(
            *pk.right_arm_get_position(self.joint_angles))
        se3_virtual_torso_in_virtual_claw = se3.inverse_matrix(se3_virtual_claw_in_virtual_torso)
        # assume hand and claw are in the same place (user did a good job) to find virtual torso estimate
        # TODO: actual rotation of controller is not same as virtual claw. correct
        #       gravity align torso frame in vrroom frame to correct error
        # TODO: actual scale is not the same. correct
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
        new_angles = pk.right_arm_set_position(self.joint_angles, new_pos, new_rot, epsilon = 0.1)
        if new_angles is not None:
            self.joint_angles = new_angles

    def visualize(self, tf_br):
        """ show the virtual arm in rviz
        """
        joint_frames = ["RShoulder", "RBicep", "RElbow", "RForeArm", "r_wrist", "RHand"]
        # get position, orientation for every joint in arm
        pos_in_torso, ori_in_torso = pk.right_arm_get_position(self.joint_angles, full_pos=True)
        time = rospy.Time.now()
        # publish tfs
        for pos, rot, frame in zip(pos_in_torso, ori_in_torso, joint_frames):
            t = TransformStamped()
            t.header.stamp = time
            t.header.frame_id = "virtualarm_RTorso"
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
        t.child_frame_id = "virtualarm_RTorso"
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
        self.virtual_arm = None
        self.current_state = "idle"

        # publishers
        self.joint_pub = rospy.Publisher("/pepper_robot/pose/joint_angles", JointAnglesWithSpeed, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # tf listener
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Subscribers
        rospy.Subscriber("/oculus/button_a_toggle", ButtonToggle, self.deadmanswitch_toggle_callback)
        rospy.Subscriber("/oculus/button_b_toggle", ButtonToggle, self.zeroswitch_toggle_callback)

        # Timer
        rospy.Timer(rospy.Duration(0.01), self.arm_controller_routine)

    def is_zero_pose_reached(self):
        return True

    def visualize_vrhand(self):
        """ publishes the static transform from controller to vrhand """
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = kRightControllerFrame
        t.child_frame_id = "oculus_right_vrhand"
        pos = se3.translation_from_matrix(se3_vrhand_in_controller)
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        q = se3.quaternion_from_matrix(se3_vrhand_in_controller)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)

    def arm_controller_routine(self, event=None):
        # show vrhand
        self.visualize_vrhand()

        # get controllers tf
        se3_controller_in_vrroom = None
        if self.current_state in ["tracking", "trackingactive", "zero"]:
            try:
                trans = self.tf_buf.lookup_transform(kVRRoomFrame, kRightControllerFrame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                return
            se3_controller_in_vrroom = se3_from_transformstamped(trans)

        # update virtual arm angles
        if self.current_state in ["tracking", "trackingactive"]:
            self.virtual_arm.update(se3_controller_in_vrroom)
            self.virtual_arm.visualize(self.tf_br)

        # get virtual arm joint angles, publish them
        if self.current_state == "trackingactive":
            pass

        if self.current_state == "zero":
            # create new temporary virtual arm and display it
            # when comfirm button is pressed the temporary arm will become permanent and joints unlocked
            # debug: visualize virtual arm before confirm button is pressed
            self.virtual_arm = VirtualArm()
            self.virtual_arm.initialize_from_zero_pose_forward_kinematics(se3_controller_in_vrroom, self.tf_br)
            self.virtual_arm.visualize(self.tf_br)


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
