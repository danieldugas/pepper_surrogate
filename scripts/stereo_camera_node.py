#!/usr/bin/env python

# original code from h3ct0r/openhmd_ros

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from image_distort import ImageLenseDistort
import numpy as np


class StereoCameraNode:

    def __init__(self, verbose=0):
        self.stereo_publisher = None
        self.cv_bridge = None
        self.r_img = None
        self.l_img = None
        self.minimap = None
        self.lense_distort = ImageLenseDistort()
        self.verbose = verbose

        rospy.init_node("stereo_cameras")

        # Subscribe for two "eyes"
        rospy.Subscriber("/camera/infra2/image_rect_raw", Image, self.r_img_callback)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.l_img_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.minimap_callback)

        self.stereo_publisher = rospy.Publisher("/openhmd/stereo", Image, queue_size=1000)
        self.cv_bridge = CvBridge()

        rospy.Timer(rospy.Duration(0.01), self.mainloop)

    def r_img_callback(self, img):
        self.r_img = img

    def l_img_callback(self, img):
        self.l_img = img

    def minimap_callback(self, img):
        self.minimap = img

    def mainloop(self, event=None):
        if self.l_img is not None and self.r_img is not None:
            dt = (self.l_img.header.stamp - self.r_img.header.stamp)
            if self.verbose > 0:
                print(dt)
            cv_right_image = self.cv_bridge.imgmsg_to_cv2(self.r_img, desired_encoding="mono8")
            cv_left_image = self.cv_bridge.imgmsg_to_cv2(self.l_img, desired_encoding="mono8")

            cv_right_image = self.lense_distort.process_frame(cv_right_image)
            cv_left_image = self.lense_distort.process_frame(cv_left_image)

            # add minimap
            if self.minimap is not None:
                cv_minimap = self.cv_bridge.imgmsg_to_cv2(self.minimap, desired_encoding="mono8")
                cv_right_image[100:64+100, 100:64+100] = cv_minimap[::cv_minimap.shape[0] / 64, ::cv_minimap.shape[1] / 64][:64, :64]
                cv_left_image[100:64+100, 20+100:64+20+100] = cv_minimap[::cv_minimap.shape[0] / 64, ::cv_minimap.shape[1] / 64][:64, :64]

            cv_stereo_image = np.append(cv_left_image, cv_right_image, axis=1)
            stereo_image = self.cv_bridge.cv2_to_imgmsg(cv_stereo_image, encoding="mono8")

            self.stereo_publisher.publish(stereo_image)

if __name__ == "__main__":
    node = StereoCameraNode(verbose=0)
    rospy.spin()
