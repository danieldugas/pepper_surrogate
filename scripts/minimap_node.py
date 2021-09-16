#!/usr/bin/env python
from geometry_msgs.msg import PoseStamped
from map2d_ros_tools import ReferenceMapAndLocalizationManager
import numpy as np
from pose2d import Pose2D, apply_tf, apply_tf_to_pose, inverse_pose2d
from pyniel.numpy_tools.indexing import as_idx_array
import rospy
import tf
from tf2_ros import TransformException
from CMap2D import CMap2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

FASTMARCH = True

class MinimapNode:

    def __init__(self, verbose=0):
        rospy.init_node('minimap_node')
        # parameters
        self.verbose = verbose
        # consts
        self.kNavGoalTopic = "/move_base_simple/goal"
        self.kRobotFrame = "base_footprint"
        self.kRobotRadius = rospy.get_param("/robot_radius", 0.3)
        self.kMinimapWidthPx = 64
        self.kMinimapHeightPx = 64
        self.kMinimapRadiusM = 10
        # vars
        self.goal_xy_in_refmap = None
        self.cv_bridge = CvBridge()
        # publishers
        self.image_publisher = rospy.Publisher("/pepper_surrogate/minimap", Image, queue_size=1)
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        self.tf_timeout = rospy.Duration(1.)
        # Localization Manager
        mapname = rospy.get_param("~reference_map_name", "map")
        mapframe = rospy.get_param("~reference_map_frame", "reference_map")
        mapfolder = rospy.get_param("~reference_map_folder", "~/maps")
        map_downsampling_passes = rospy.get_param("~reference_map_downsampling_passes", 3)
        def refmap_update_callback(self_):
            self_.map_8ds = self_.map_
            for _ in range(map_downsampling_passes):
                self_.map_8ds = self_.map_8ds.as_coarse_map2d()
            self_.map_8ds_sdf = self_.map_8ds.as_sdf()
        self.refmap_manager = ReferenceMapAndLocalizationManager(
            mapfolder, mapname, mapframe, self.kRobotFrame,
            refmap_update_callback=refmap_update_callback,
        )
        self.refmap_manager.map_8ds = None
        # callback
        rospy.Subscriber(self.kNavGoalTopic, PoseStamped, self.global_goal_callback, queue_size=1)
        # Timers
        rospy.Timer(rospy.Duration(0.1), self.mainloop)

    def global_goal_callback(self, msg): # x y is in the global map frame
        rospy.loginfo("set_goal message received")
        set_goal_msg = msg
        try:
            time = rospy.Time.now()
            tf_info = [self.refmap_manager.kRefMapFrame, set_goal_msg.header.frame_id, time]
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_msg_in_refmap = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            print("[{}.{}] tf to refmap frame for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return
        pose2d_msg_in_refmap = Pose2D(tf_msg_in_refmap)
        self.goal_xy_in_refmap = apply_tf(np.array([set_goal_msg.pose.position.x,
                                                    set_goal_msg.pose.position.y]), pose2d_msg_in_refmap)

    def mainloop(self, event=None):
        R = np.zeros((self.kMinimapWidthPx, self.kMinimapHeightPx))
        G = np.zeros((self.kMinimapWidthPx, self.kMinimapHeightPx))
        B = np.zeros((self.kMinimapWidthPx, self.kMinimapHeightPx))
        A = np.ones((self.kMinimapWidthPx, self.kMinimapHeightPx))
        if self.refmap_manager.tf_frame_in_refmap is None:
            # white noise with text "location unknown"
            O_ = np.random.uniform(size=R.shape)
            R = O_ * 1.
            G = O_ * 1.
            B = O_ * 1.
        elif self.refmap_manager.map_8ds is None:
            # white noise with text "map unknown"
            O_ = np.random.uniform(size=R.shape)
            R = O_ * 1.
            G = O_ * 1.
            B = O_ * 1.
        else:
            # everything in minimap frame (minimap is Map2D with base_footprint as origin)
            resolution = self.kMinimapRadiusM * 2. / self.kMinimapWidthPx
            self.minimap2d = CMap2D()
            self.minimap2d.unserialize({
                "occupancy": np.zeros((self.kMinimapWidthPx, self.kMinimapHeightPx), dtype=np.float32),
                "occupancy_shape0": self.kMinimapWidthPx,
                "occupancy_shape1": self.kMinimapHeightPx,
                "resolution_": resolution,
                "_thresh_occupied": 0.9,
                "thresh_free": 0.1,
                "HUGE_": 1e10,
                # 0, 0 coordinate in the middle
                "origin": np.array([- resolution * self.kMinimapWidthPx / 2. ,
                                    - resolution * self.kMinimapHeightPx / 2.], dtype=np.float32),
            })
            # transform between minimap and base_footprint
            pose2d_base_footprint_in_minimap = np.array([0, 0, np.pi])
            # transform between minimap and refmap
            # refmap -> base_footprint -> minimap
            pose2d_base_footprint_in_refmap = Pose2D(self.refmap_manager.tf_frame_in_refmap)
            pose2d_refmap_in_base_footprint = inverse_pose2d(pose2d_base_footprint_in_refmap)
            pose2d_refmap_in_minimap = apply_tf_to_pose(
                pose2d_refmap_in_base_footprint, pose2d_base_footprint_in_minimap)
            pose2d_minimap_in_refmap = inverse_pose2d(pose2d_refmap_in_minimap)
            # create miniature map
            pixel_ij_in_minimap = as_idx_array(self.minimap2d.occupancy(), axis='all').reshape((-1, 2))
            pixel_xy_in_minimap = self.minimap2d.ij_to_xy(pixel_ij_in_minimap)
            pixel_xy_in_refmap = np.ascontiguousarray(apply_tf(pixel_xy_in_minimap, pose2d_minimap_in_refmap))
            pixel_ij_in_refmap = self.refmap_manager.map_8ds.xy_to_ij(pixel_xy_in_refmap,
                                                                      clip_if_outside=True)
            try:
                pixel_values = self.refmap_manager.map_8ds.occupancy()[(pixel_ij_in_refmap[:, 0],
                                                                       pixel_ij_in_refmap[:, 1])]
                pixel_values = pixel_values.reshape((self.kMinimapWidthPx, self.kMinimapHeightPx))
                pixel_values[pixel_values < 0] = 0.5
                if FASTMARCH:
                    fm = self.refmap_manager.map_8ds.fastmarch(
                        self.refmap_manager.map_8ds.xy_to_ij(pose2d_base_footprint_in_refmap[None, :2])[0])
                    fastmarch_values = fm[(pixel_ij_in_refmap[:, 0],
                                           pixel_ij_in_refmap[:, 1])].reshape(
                                               (self.kMinimapWidthPx, self.kMinimapHeightPx))
            except IndexError as e:
                print(e)
                return
            self.minimap2d._occupancy = pixel_values
            inv_occ = (1. - self.minimap2d.occupancy())
            R = inv_occ
            G = R * 1.
            B = R * 1.
            A = np.ones_like(R)
            if FASTMARCH:
                from matplotlib import pyplot as plt
                normalizer = np.nanmax(fm[fm != np.inf])
                colors = plt.cm.viridis(fastmarch_values / normalizer)
                R = colors[:, :, 0]
                G = colors[:, :, 1]
                B = colors[:, :, 2]
                R[pixel_values > 0.2] = 0
                G[pixel_values > 0.2] = 0
                B[pixel_values > 0.2] = 0
                A[pixel_values > 0.2] = 0

        # arrow stencil
        center = np.array([self.kMinimapWidthPx / 2, self.kMinimapHeightPx / 2], dtype=int)
        rel = np.array([[0, 1], [-1, 0], [0, 0], [-1, -1], [0, -1], [0, -2]])
        stencil = (center + rel)
        stencil = (stencil[:, 0], stencil[:, 1])
        R[stencil] = 0
        G[stencil] = 1
        B[stencil] = 0.3

        # goal stencil
        if self.goal_xy_in_refmap is not None:
            goal_xy_in_minimap = np.squeeze(apply_tf(self.goal_xy_in_refmap[None, :],
                                                     pose2d_refmap_in_minimap))
            goal_ij_in_minimap = np.squeeze(self.minimap2d.xy_to_ij(goal_xy_in_minimap[None, :],
                                                                    clip_if_outside=True))
            gc = goal_ij_in_minimap
            from pyniel.numpy_tools.circular_index import make_circular_index
            reli, relj = make_circular_index(2)
            rel = np.concatenate([reli[:,None], relj[:,None]], axis=1)
            stencil = gc + rel
            stencil = stencil[np.where(self.minimap2d.is_inside_ij(stencil.astype(np.float32)))]
            stencil = (stencil[:, 0], stencil[:, 1])
            R[stencil] = 1
            G[stencil] = 1
            B[stencil] = 0

        cv_minimap = np.zeros((self.kMinimapWidthPx, self.kMinimapHeightPx, 4), dtype=np.uint8)
        cv_minimap[:, :, 0] = (R * 255).astype(np.uint8)
        cv_minimap[:, :, 1] = (G * 255).astype(np.uint8)
        cv_minimap[:, :, 2] = (B * 255).astype(np.uint8)
        cv_minimap[:, :, 3] = (A * 255).astype(np.uint8)

        minimap_msg = self.cv_bridge.cv2_to_imgmsg(cv_minimap, encoding="rgba8")
        self.image_publisher.publish(minimap_msg)


if __name__ == "__main__":
    node = MinimapNode(verbose=0)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt - shutting down.")
        rospy.signal_shutdown('KeyboardInterrupt')
