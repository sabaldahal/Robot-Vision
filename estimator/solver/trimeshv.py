import cv2 as cv
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R_s
from rvec_analyzer import *


class SolveVectorTrimesh:
    def __init__(self):
        self.mesh = trimesh.load("./estimator/model/spacecraft.obj", force='mesh')

    def solve(self, rvec, tvec):

        
        #scene axis correction using rotation 
        Rscene = trimesh.transformations.rotation_matrix(
            angle=np.deg2rad(-180),
            direction=[1, 0, 0],  # X-axis
            point=[0, 0, 0]
        )

        #rvec to rotation in trimesh
        Ro, _ = cv.Rodrigues(rvec)

        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            R = np.eye(3)
        else:
            axis = (rvec / theta).flatten()
            print("axis", axis)
            print("angle", np.rad2deg(theta))
            x, y, z = axis
            K = np.array([[ 0, -z,  y],
                        [ z,  0, -x],
                        [-y,  x,  0]])
            R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            #analysis
            analyzer = Analyzer()
            analyzer.axisAngleAnalysis(axis, theta)
            analyzer.getRvecAccuracy(rvec)

        
        noT = np.eye(4)
        noT[:3, :3] = R 
        Translation = tvec.flatten()
        noT[:3, 3] = Translation
        print("Ro", Ro)
        print("noT", noT)
        print("rvec", rvec)
        print("tvec", tvec)
        print("tvec flattened", Translation)
        scene = trimesh.Scene(self.mesh) 
        self.mesh.apply_transform(noT)
        scene.apply_transform(Rscene)  
        
        scene.show()

