import cv2 as cv
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R_s
from rvec_analyzer import *
import threading

#this will result in segmentation fault.
#find a non-blocking solution so that new transformation can be dynamically passed during runtime
class SolveVectorTrimesh:
    def __init__(self):
        self.mesh = trimesh.load("./estimator/model/spacecraft.obj", force='mesh')
        self.scene = trimesh.Scene(self.mesh)
        self._latest_transform = np.eye(4)
        self._initializeScene()

    def _initializeScene(self):        
        #scene axis correction using rotation 
        Rscene = trimesh.transformations.rotation_matrix(
            angle=np.deg2rad(-180),
            direction=[1, 0, 0],  # X-axis
            point=[0, 0, 0]
        )
        self.scene.apply_transform(Rscene)  

    def customRodriguesTransform(self, rvec):
        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            rotation = np.eye(3)
        else:
            axis = (rvec / theta).flatten()
            x, y, z = axis
            K = np.array([[ 0, -z,  y],
                        [ z,  0, -x],
                        [-y,  x,  0]])
            rotation = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
        return rotation

    def applyTransform(self, rvec, tvec):
        translation = tvec.flatten()
        T_matrix = np.eye(4)
        rotation, _ = cv.Rodrigues(rvec)
        T_matrix[:3, :3] = rotation
        T_matrix[:3, 3] = translation
        self._latest_transform = T_matrix

    def _runViewer(self):
        self.scene.show(callback = self._callback)

    def _callback(self, scene):
        if not np.allclose(self._latest_transform, np.eye(4)):
            self.mesh.apply_transform(self._latest_transform)
            self._latest_transform = np.eye(4)

    def start(self):
        thread = threading.Thread(target=self._runViewer, daemon=True)
        thread.start()
        return thread
