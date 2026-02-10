import cv2 as cv
import numpy as np
import open3d as o3d

class SolveVectorO3D:
    def __init__(self):
        self.mesh = o3d.io.read_triangle_mesh("./estimator/model/spacecraft.obj")
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()