import cv2 as cv
import numpy as np
import math

class Analyzer:

    def getRvecAccuracy(self, rvec):
        R_cv, _ = cv.Rodrigues(rvec)

        #actual orientation
        axis = np.array([0.109, -0.155, -0.982])
        angle = np.deg2rad(119)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_bl = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
        R_diff = R_cv @ R_bl.T
        cos_theta = (np.trace(R_diff) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
        angle_diff = np.arccos(cos_theta)  # in radians
        angle_diff_deg = np.degrees(angle_diff)

        rvec_diff, _ = cv.Rodrigues(R_diff)
        axis_diff = rvec_diff.flatten() / np.linalg.norm(rvec_diff)
        print("difference in angle:", angle_diff_deg)
        print("axis of misalignment difference:", axis_diff)
        
    def axisAngleAnalysis(self, axis, theta):
        axis = axis / np.linalg.norm(axis)
        
        w = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        x, y, z = axis * s

        q_wxyz = (w, x, y, z)    # scalar-first
        q_xyzw = (x, y, z, w)    # vector-first

        print("q (w,x,y,z) =", q_wxyz)
        print("q (x,y,z,w) =", q_xyzw)