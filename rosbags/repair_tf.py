import numpy as np
import rosbag
import tf.transformations as se3

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


if __name__ == "__main__":
    with rosbag.Bag('vicon_both_arms_test_corrected.bag', 'w') as outbag:
        for topic, msg, t in rosbag.Bag('vicon_both_arms_test.bag').read_messages():
            # This also replaces tf timestamps under the assumption
            # that all transforms in the message share the same timestamp
            if topic == "/tf" and msg.transforms:
                for tf in msg.transforms:
                    if tf.child_frame_id == "vicon/vicon_left_armband/vicon_left_armband":
                        v_T_ab = se3_from_transformstamped(tf)
                        ab_T_trueab = se3.euler_matrix(0, 0, np.pi)
                        v_T_trueab = np.dot(v_T_ab, ab_T_trueab)
                        q = se3.quaternion_from_matrix(v_T_trueab)
                        tf.transform.rotation.x = q[0]
                        tf.transform.rotation.y = q[1]
                        tf.transform.rotation.z = q[2]
                        tf.transform.rotation.w = q[3]
                outbag.write(topic, msg, t)
            else:
                outbag.write(topic, msg, t)
