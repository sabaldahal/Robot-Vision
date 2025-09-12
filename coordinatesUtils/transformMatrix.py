
import json
import os
import numpy as np
import bpy
from mathutils import Matrix


obj = bpy.data.objects.get('scobj')
cam = bpy.data.objects['RealSense Camera']


obj_world = obj.matrix_world
cam_world = cam.matrix_world
cam_mat_world_inv = cam_world.inverted()



# extract translation + rotation only (no scale)
cam_rot = cam_world.to_quaternion().to_matrix().to_4x4()   # pure rotation
cam_trans = Matrix.Translation(cam_world.to_translation()) # pure translation

# rebuild camera world transform (rotation + translation, no scale)
cam_world_rigid = cam_trans @ cam_rot

# invert and apply to object
obj_in_cam = cam_world_rigid.inverted() @ obj_world




# wd = os.getcwd()
# file = os.path.join(wd, "orientation_matrix.txt")

matrix_np = np.array(obj_in_cam)
# np.savetxt(file, matrix_np)


print("Object in camera coordinates (rigid, preserves rotation):")
print(obj_in_cam)
print("Translation (meters):", obj_in_cam.to_translation())
print("Rotation (quaternion):", obj_in_cam.to_quaternion())

# convert to numpy
M = np.array(obj_in_cam)

# save to txt with 7 decimals'
a = "/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/estimator/solver"
file = os.path.join(a, "matrix54.txt")

np.savetxt(file, M, fmt="%.7f")
print("Saved matrix to object_in_camera.txt")
