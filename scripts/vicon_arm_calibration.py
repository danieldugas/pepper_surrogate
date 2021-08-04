import numpy as np
import tf.transformations as se3

class ViconCalibrationResult(object):
    se3_left_shoulder_in_torso = None
    se3_left_shoulder_in_left_armband = se3.translation_matrix(np.array([-0.2, 0., 0.]))
    se3_left_elbow_in_left_armband = se3.translation_matrix(np.array([0.05, 0., 0.]))
    se3_left_wrist_in_left_wristband = se3.identity_matrix()
    se3_right_shoulder_in_torso = None
    se3_right_shoulder_in_right_armband = se3.translation_matrix(np.array([-0.2, 0., 0.]))
    se3_right_elbow_in_right_armband = se3.translation_matrix(np.array([0.05, 0., 0.]))
    se3_right_wrist_in_right_wristband = se3.identity_matrix()
    is_armband_orientation_calibrated = False

    def __init__(self):
        pass

def load_vicon_calibration(file):
    vicon_calib = ViconCalibrationResult()
    # TODO
    # ...
    return vicon_calib

class ViconCalibrationNode(object):
    def __init__(self):
        pass
