#!/usr/bin/env python

# original code from h3ct0r/openhmd_ros

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from cv_bridge import CvBridge
import cv2
from image_distort import ImageLenseDistort
import numpy as np


class StereoCameraNode:

    def __init__(self, verbose=0):
        rospy.init_node("stereo_cameras")
        # publishers
        self.stereo_publisher = rospy.Publisher("/pepper_surrogate/stereo", Image, queue_size=1000)
        # tools
        self.cv_bridge = CvBridge()
        self.lense_distort = ImageLenseDistort()
        # parameters
        self.verbose = verbose
        # variables
        self.r_img = None
        self.l_img = None
        self.minimap = None
        self.headyaw = None
        self.cmd_vel_enabled = True

        # Subscribe for two "eyes"
        rospy.Subscriber("/camera/infra2/image_rect_raw", Image, self.r_img_callback)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.l_img_callback)
        rospy.Subscriber("/pepper_surrogate/minimap", Image, self.minimap_callback)
        rospy.Subscriber("/pepper_robot/pose/joint_angles", JointAnglesWithSpeed, self.headyaw_callback)
        rospy.Subscriber("/oculus/cmd_vel_enabled", Bool, self.cmd_vel_enabled_callback)

        rospy.Timer(rospy.Duration(0.01), self.mainloop)

    def r_img_callback(self, img):
        self.r_img = img

    def l_img_callback(self, img):
        self.l_img = img

    def minimap_callback(self, img):
        self.minimap = img

    def headyaw_callback(self, msg):
        for i in range(len(msg.joint_names)):
            if msg.joint_names[i] == "HeadYaw":
                self.headyaw = msg.joint_angles[i]

    def cmd_vel_enabled_callback(self, msg):
        self.cmd_vel_enabled = msg.data

    def mainloop(self, event=None):
        if self.l_img is not None and self.r_img is not None:
            dt = (self.l_img.header.stamp - self.r_img.header.stamp)
            if self.verbose > 0:
                print(dt)
            cv_right_image = self.cv_bridge.imgmsg_to_cv2(self.r_img, desired_encoding="mono8")
            cv_left_image = self.cv_bridge.imgmsg_to_cv2(self.l_img, desired_encoding="mono8")

            cv_right_image = self.lense_distort.process_frame(cv_right_image)
            cv_left_image = self.lense_distort.process_frame(cv_left_image)

            # add color channels
            a = np.ones_like(cv_right_image)
            cv_right_image = cv2.merge((cv_right_image, cv_right_image, cv_right_image, a))
            cv_left_image = cv2.merge((cv_left_image, cv_left_image, cv_left_image, a))

            # add minimap
            if self.minimap is not None:
                cv_minimap = self.cv_bridge.imgmsg_to_cv2(self.minimap, desired_encoding="rgba8")
                cv_right_image[100:64+100, 100:64+100] = cv_minimap
                cv_left_image[100:64+100, 20+100:64+20+100] = cv_minimap

            # add cmd_vel_enabled indicator
            if not self.cmd_vel_enabled:
                cv_right_image[200:64+200, 100:64+100, :] = 255
                cv_right_image[200:64+200, 100:64+100, 1:3] = 0
                cv_left_image[200:64+200, 20+100:64+20+100, :] = 255
                cv_left_image[200:64+200, 20+100:64+20+100, 1:3] = 0

            # add headyaw indicator
            if self.headyaw is not None:
                minyaw = -0.5
                maxyaw = 0.5
                normalizedyaw = (self.headyaw - minyaw) / (maxyaw - minyaw) # [0,1]
                normalizedyaw = np.clip(normalizedyaw, 0, 1)
                W = cv_right_image.shape[1]-1
                pxyawr = int(normalizedyaw * W)
                pxyawr = np.clip(pxyawr, 1, W)
                pxrefr = int(0.5 * W)
                pxyawl = pxyawr + 20
                pxyawl = np.clip(pxyawl, 1, W)
                pxrefl = int(0.5 * W) + 20
                cv_left_image[ -50:, pxyawl, :] = 255
                cv_left_image[ -50:, pxyawl-1, :] = 0
                cv_right_image[-50:, pxyawr, :] = 255
                cv_right_image[-50:, pxyawr-1, :] = 0
                cv_left_image[ -50:-30, pxrefl-3, :] = 255
                cv_left_image[ -50:-30, pxrefl-4, :] = 0
                cv_right_image[-50:-30, pxrefr-3, :] = 255
                cv_right_image[-50:-30, pxrefr-4, :] = 0
                cv_left_image[ -50:-30, pxrefl+3, :] = 255
                cv_left_image[ -50:-30, pxrefl+2, :] = 0
                cv_right_image[-50:-30, pxrefr+3, :] = 255
                cv_right_image[-50:-30, pxrefr+2, :] = 0

            cv_stereo_image = np.append(cv_left_image, cv_right_image, axis=1)
            stereo_image = self.cv_bridge.cv2_to_imgmsg(cv_stereo_image, encoding="rgba8")

            self.stereo_publisher.publish(stereo_image)

if __name__ == "__main__":
    node = StereoCameraNode(verbose=0)
    rospy.spin()
