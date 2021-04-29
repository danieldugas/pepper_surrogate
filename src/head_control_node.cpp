#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <naoqi_bridge_msgs/JointAnglesWithSpeed.h>

#include <openhmd.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void ohmd_sleep(double);

// gets float values from the device and prints them
void print_infof(ohmd_device* hmd, const char* name, int len, ohmd_float_value val)
{
  float f[16];
  assert(len <= 16);
  ohmd_device_getf(hmd, val, f);
  printf("%-25s", name);
  for(int i = 0; i < len; i++)
    printf("%f ", f[i]);
  printf("\n");
}

// gets int values from the device and prints them
void print_infoi(ohmd_device* hmd, const char* name, int len, ohmd_int_value val)
{
  int iv[16];
  assert(len <= 16);
  ohmd_device_geti(hmd, val, iv);
  printf("%-25s", name);
  for(int i = 0; i < len; i++)
    printf("%d ", iv[i]);
  printf("\n");
}

namespace head_control_node {

/// \brief A circular array based on std::vector
template <class T>
class CircularArray {
  public:
    CircularArray(std::initializer_list<T> l) : data_(l) {};
    T getNextItem() {
      T result = data_.at(current_pos_);
      if ( ++current_pos_ >= data_.size() ) {
        current_pos_ = 0;
      }
      return result;
    }

  private:
    std::size_t current_pos_ = 0;
    const std::vector<T> data_;
}; // class CircularArray

class OculusHeadController {

  public:
    explicit OculusHeadController(ros::NodeHandle& n) : nh_(n), tfListener_(tfBuffer_) {
      int device_idx = 0;
      
      if (init_hmd(device_idx) != 0) {
        ROS_ERROR_STREAM("FAILED TO INITIALIZE HMD DEVICE");
        return;
      }

      // Topic names.
      const std::string kPepperPoseTopic = "/pepper_robot/pose/joint_angles";
      kFixedFrame = "odom";
      kHMDFrame = "oculus";
      kBaseLinkFrame = "base_footprint";

      // Publishers and subscribers.
      pose_pub_ = nh_.advertise<naoqi_bridge_msgs::JointAnglesWithSpeed>(kPepperPoseTopic, 1);

      // Initialize times.
      pose_pub_last_publish_time_ = ros::Time::now();



      // Timer callbacks
      pose_timer_ = nh_.createTimer(ros::Duration(0.01), &OculusHeadController::timerCallback, this);
      get_tf_timer_ = nh_.createTimer(ros::Duration(0.01), &OculusHeadController::gettfCallback, this);

    }
    ~OculusHeadController() {}

  protected:
    int init_hmd(int device_idx) {
      ohmd_require_version(0, 3, 0);

      int major, minor, patch;
      ohmd_get_version(&major, &minor, &patch);

      printf("OpenHMD version: %d.%d.%d\n", major, minor, patch);

      ctx_ = ohmd_ctx_create();

      // Probe for devices
      int num_devices = ohmd_ctx_probe(ctx_);
      if(num_devices < 0){
        printf("failed to probe devices: %s\n", ohmd_ctx_get_error(ctx_));
        return -1;
      }

      printf("num devices: %d\n\n", num_devices);

      // Print device information
      for(int i = 0; i < num_devices; i++){
        int device_class = 0, device_flags = 0;
        const char* device_class_s[] = {"HMD", "Controller", "Generic Tracker", "Unknown"};

        ohmd_list_geti(ctx_, i, OHMD_DEVICE_CLASS, &device_class);
        ohmd_list_geti(ctx_, i, OHMD_DEVICE_FLAGS, &device_flags);

        printf("device %d\n", i);
        printf("  vendor:  %s\n", ohmd_list_gets(ctx_, i, OHMD_VENDOR));
        printf("  product: %s\n", ohmd_list_gets(ctx_, i, OHMD_PRODUCT));
        printf("  path:    %s\n", ohmd_list_gets(ctx_, i, OHMD_PATH));
        printf("  class:   %s\n", device_class_s[device_class > OHMD_DEVICE_CLASS_GENERIC_TRACKER ? 4 : device_class]);
        printf("  flags:   %02x\n",  device_flags);
        printf("    null device:         %s\n", device_flags & OHMD_DEVICE_FLAGS_NULL_DEVICE ? "yes" : "no");
        printf("    rotational tracking: %s\n", device_flags & OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING ? "yes" : "no");
        printf("    positional tracking: %s\n", device_flags & OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING ? "yes" : "no");
        printf("    left controller:     %s\n", device_flags & OHMD_DEVICE_FLAGS_LEFT_CONTROLLER ? "yes" : "no");
        printf("    right controller:    %s\n\n", device_flags & OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER ? "yes" : "no");
      }

      // Open specified device idx or 0 (default) if nothing specified
      printf("opening device: %d\n", device_idx);
      hmd_ = ohmd_list_open_device(ctx_, device_idx);
      
      if(!hmd_){
        printf("failed to open device: %s\n", ohmd_ctx_get_error(ctx_));
        return -1;
      }

      // Print hardware information for the opened device
      int ivals[2];
      ohmd_device_geti(hmd_, OHMD_SCREEN_HORIZONTAL_RESOLUTION, ivals);
      ohmd_device_geti(hmd_, OHMD_SCREEN_VERTICAL_RESOLUTION, ivals + 1);
      printf("resolution:              %i x %i\n", ivals[0], ivals[1]);

      print_infof(hmd_, "hsize:",            1, OHMD_SCREEN_HORIZONTAL_SIZE);
      print_infof(hmd_, "vsize:",            1, OHMD_SCREEN_VERTICAL_SIZE);
      print_infof(hmd_, "lens separation:",  1, OHMD_LENS_HORIZONTAL_SEPARATION);
      print_infof(hmd_, "lens vcenter:",     1, OHMD_LENS_VERTICAL_POSITION);
      print_infof(hmd_, "left eye fov:",     1, OHMD_LEFT_EYE_FOV);
      print_infof(hmd_, "right eye fov:",    1, OHMD_RIGHT_EYE_FOV);
      print_infof(hmd_, "left eye aspect:",  1, OHMD_LEFT_EYE_ASPECT_RATIO);
      print_infof(hmd_, "right eye aspect:", 1, OHMD_RIGHT_EYE_ASPECT_RATIO);
      print_infof(hmd_, "distortion k:",     6, OHMD_DISTORTION_K);
      
      print_infoi(hmd_, "control count:   ", 1, OHMD_CONTROL_COUNT);

      ohmd_device_geti(hmd_, OHMD_CONTROL_COUNT, &hmd_control_count_);

      const char* controls_fn_str[] = { "generic", "trigger", "trigger_click", "squeeze", "menu", "home",
        "analog-x", "analog-y", "anlog_press", "button-a", "button-b", "button-x", "button-y",
        "volume-up", "volume-down", "mic-mute"};

      const char* controls_type_str[] = {"digital", "analog"};

      int controls_fn[64];
      int controls_types[64];

      ohmd_device_geti(hmd_, OHMD_CONTROLS_HINTS, controls_fn);
      ohmd_device_geti(hmd_, OHMD_CONTROLS_TYPES, controls_types);
      
      printf("%-25s", "controls:");
      for(int i = 0; i < hmd_control_count_; i++){
        printf("%s (%s)%s", controls_fn_str[controls_fn[i]], controls_type_str[controls_types[i]], i == hmd_control_count_ - 1 ? "" : ", ");
      }

      printf("\n\n");

      return 0;
    }

    void gettfCallback(const ros::TimerEvent& e) {
      try{
        pepper_in_world_ = tfBuffer_.lookupTransform(kFixedFrame, kBaseLinkFrame, ros::Time(0));
      }
      catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
      }
    }

    void timerCallback(const ros::TimerEvent& e) {

      ohmd_ctx_update(ctx_);

      // this can be used to set a different zero point
      // for rotation and position, but is not required.
      //float zero[] = {.0, .0, .0, 1};
      //ohmd_device_setf(hmd_, OHMD_ROTATION_QUAT, zero);
      //ohmd_device_setf(hmd_, OHMD_POSITION_VECTOR, zero);

      // get rotation and position
      print_infof(hmd_, "rotation quat:", 4, OHMD_ROTATION_QUAT);
      print_infof(hmd_, "position vec: ", 3, OHMD_POSITION_VECTOR);

      float f[16];
      ohmd_device_getf(hmd_, OHMD_ROTATION_QUAT, f);
      tf2::Quaternion q(
      f[0],
      f[1],
      f[2],
      f[3]
      );


      tf2::Quaternion qtilt;
      qtilt.setRPY(3.14159/2., 0.0, 0.0);


      // Rotate the previous pose by 90 about X
      // Then 90 about Z
      tf2::Quaternion qrotx;
      tf2::Quaternion qrotz;
      qrotx.setRPY(-3.14159/2., 0.0, 0.0);
      qrotz.setRPY(0.0, 0.0, 3.14159/2.);
      // This is the same as creating a world to qtilt tf with qtilt as rot, (rotates the actual oculus frame)
      // then a qtilt to oculus tf with q * qrotx * qrotz as tf (rotates the axes around the oculus)
      q = qtilt * q * qrotx * qrotz;

      geometry_msgs::TransformStamped transformStamped;
      transformStamped.header.stamp = ros::Time::now();
      transformStamped.header.frame_id = kFixedFrame;
      transformStamped.child_frame_id = kHMDFrame;
      transformStamped.transform.translation.x = 0.0;
      transformStamped.transform.translation.y = 0.0;
      transformStamped.transform.translation.z = 1.0;
      transformStamped.transform.rotation.x = q.x();
      transformStamped.transform.rotation.y = q.y();
      transformStamped.transform.rotation.z = q.z();
      transformStamped.transform.rotation.w = q.w();
      br_.sendTransform(transformStamped);

      // oculus pose as euler angles in world frame
      double oculus_roll, oculus_pitch, oculus_yaw;
      tf2::Matrix3x3(q).getRPY(oculus_roll, oculus_pitch, oculus_yaw);

      // pepper yaw in world frame (a.k.a oculus in base_footprint tf - rot only)
      double pepper_roll, pepper_pitch, pepper_yaw;
      tf2::Quaternion qpepper_in_world(
          pepper_in_world_.transform.rotation.x,
          pepper_in_world_.transform.rotation.y,
          pepper_in_world_.transform.rotation.z,
          pepper_in_world_.transform.rotation.w);
      tf2::Matrix3x3(qpepper_in_world).getRPY(pepper_roll, pepper_pitch, pepper_yaw);
      // we only care about yaw (assume base_footprint is horizontal)
      // TODO: check that base_footprint and world z axis are aligned?
      double relative_yaw = oculus_yaw - pepper_yaw;

      // Control pepper head to oculus pose
      // Pepper head has no roll axis, use hip?
      // how do we track pepper's head direction?
      const static float kMaxHeadMoveAngleRad = 1.;
      const static float kMaxJointSpeedRadPerS = 0.2;
      const static float kMinHeadYawRad = -2.0857;
      const static float kMaxHeadYawRad =  2.0857;
      const static float kMinHeadPitchRad = -0.7068;
      const static float kMaxHeadPitchRad =  0.6371;
      const static std::string kHeadYawJointName = "HeadYaw";
      const static std::string kHeadPitchJointName = "HeadPitch";
      const static ros::Duration kMaxPosePubRate(0.1);
      // Limit the maximum rate publishing rate for pose control
      if ( ( ros::Time::now() - pose_pub_last_publish_time_ ) > kMaxPosePubRate )
      {
        naoqi_bridge_msgs::JointAnglesWithSpeed joint_angles_msg;
        joint_angles_msg.speed = kMaxJointSpeedRadPerS * 0.5;
        joint_angles_msg.joint_names.push_back(kHeadYawJointName);
        joint_angles_msg.joint_angles.push_back(relative_yaw);
        joint_angles_msg.joint_names.push_back(kHeadPitchJointName);
        joint_angles_msg.joint_angles.push_back(oculus_pitch);
        pose_pub_.publish(joint_angles_msg);
        pose_pub_last_publish_time_ = ros::Time::now();
      }

      // read controls
      if (hmd_control_count_) {
        float control_state[256];
        ohmd_device_getf(hmd_, OHMD_CONTROLS_STATE, control_state);

        printf("%-25s", "controls state:");
        for(int i = 0; i < hmd_control_count_; i++)
        {
          printf("%f ", control_state[i]);
        }
      }
      puts("");
        
//       ohmd_sleep(.01);
    }

  private:
    ros::NodeHandle& nh_;
    ros::Timer pose_timer_;
    ros::Timer get_tf_timer_;
    ros::Publisher pose_pub_;
    ros::Time pose_pub_last_publish_time_;
    tf2_ros::TransformBroadcaster br_;
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
    geometry_msgs::TransformStamped pepper_in_world_;
    std::string kBaseLinkFrame;
    std::string kFixedFrame;
    std::string kHMDFrame;

    ohmd_context* ctx_;
    ohmd_device* hmd_;
    int hmd_control_count_;

}; // class OculusHeadController

} // namespace head_control_node

using namespace head_control_node;

int main(int argc, char **argv) {

//   oldmain(argc, argv);

  ros::init(argc, argv, "head_control_node");
  ros::NodeHandle n;
  OculusHeadController head_controller(n);

  try {
    ros::spin();
  }
  catch (const std::exception& e) {
    ROS_ERROR_STREAM("Exception: " << e.what());
    return 1;
  }
  catch (...) {
    ROS_ERROR_STREAM("Unknown Exception.");
    return 1;
  }

  return 0;
}
