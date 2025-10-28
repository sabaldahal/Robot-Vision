import cv2 as cv
import numpy as np
import math

class Analyzer:

    def getRotationError(self, R1, R2):
        R_diff = R2 @ R1.T
        trace = np.trace(R_diff)
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # clip for numerical stability
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def getTranslationError(self, t1, t2):
        t1 = np.array(t1)
        t2 = np.array(t2)
        error = np.linalg.norm(t1 - t2)  # Euclidean distance
        return error