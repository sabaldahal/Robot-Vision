import bpy
import json
import os
import numpy as np


obj = bpy.data.objects.get('scobj')
cam = bpy.data.objects['RealSense Camera']


obj_mat_world = obj.matrix_world
cam_mat_world = cam.matrix_world
cam_mat_world_inv = cam_mat_world.inverted()

obj_in_cam_co = cam_mat_world_inv @ obj_mat_world

wd = os.getcwd()
file = os.path.join(wd, "orientation_matrix.txt")

matrix_np = np.array(obj_in_cam_co)
np.savetxt(file, matrix_np)
