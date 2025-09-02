import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np

import sys
sys.path.append('./estimator/solver')

from o3dv import *
from trimeshv import *


if not hasattr(np, "infty"):
    np.infty = np.inf

import argparse

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-i', '--image', type=str, default='48.png', help='Path to the input image')


args = parser.parse_args()
img_path_arg = f'./estimator/inferenceImages/{args.image}'





obj_points = []
img_points = []

#ground truth keypoints
coords_file = "./estimator/model/coords.json"
keypointsArr = []
with open (coords_file, "r") as f:
    keypointsArr = json.load(f)

for k in keypointsArr:
    obj_points.append(k['location'])


fx = 915.5166015625
fy = 915.607421875
cx = 629.287109375
cy = 356.802307128906


cam_mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

model_path = "./estimator/weights/best.pt"

img_path = img_path_arg
frame = cv.imread(img_path)

model = YOLO(model_path)

result = model(img_path)[0]
keypoints = result.keypoints.xy.cpu().numpy()
bboxes = result.boxes.xyxy.cpu().numpy()

#detected keypoints
for kps in keypoints:
    for x, y in kps:
        img_points.append([x,y])

obj_points = np.array(obj_points, dtype=np.float32)
img_points  = np.array(img_points,  dtype=np.float32)





def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.split()
                # Convert x, y, z coordinates to floats
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def load_obj_faces(filepath):
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()
                face_indices = []
                for p in parts[1:]:
                    # Handle cases like "1", "1/2", or "1/2/3"
                    vertex_index = int(p.split('/')[0]) - 1  # Subtract 1 for 0-based indexing
                    face_indices.append(vertex_index)
                faces.append(face_indices)
    return np.array(faces, dtype=int)

mesh_file = "./estimator/model/test.obj"
vertices_array = load_obj_vertices(mesh_file)
faces_array = load_obj_faces(mesh_file)


success, rvec, tvec = cv.solvePnP(
    obj_points, 
    img_points, 
    cam_mat, 
    dist_coeffs
)

import numpy as np

def project_points_numpy(objectPoints, rvec, tvec, cam_mat):
    """
    Project 3D points to 2D using OpenCV-style pinhole camera model.
    No distortion applied.

    Parameters:
        objectPoints : Nx3 array of 3D points
        rvec : 3x1 rotation vector (Rodrigues)
        tvec : 3x1 translation vector
        cam_mat : 3x3 camera intrinsic matrix

    Returns:
        Nx2 array of 2D points in pixel coordinates
    """
    objectPoints = np.asarray(objectPoints).reshape(-1,3)
    tvec = np.asarray(tvec).reshape(3,1)

    # Convert rvec to rotation matrix
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        R = np.eye(3)
    else:
        axis = (rvec / theta).flatten()
        x, y, z = axis
        K = np.array([[ 0, -z,  y],
                      [ z,  0, -x],
                      [-y,  x,  0]])
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

    print('R from custom project', R)
    # Transform points to camera coordinates
    points_cam = (R @ objectPoints.T) + tvec  # shape 3xN

    # Perspective division
    x = points_cam[0,:] / points_cam[2,:]
    y = points_cam[1,:] / points_cam[2,:]

    # Apply camera intrinsics
    fx, fy = cam_mat[0,0], cam_mat[1,1]
    cx, cy = cam_mat[0,2], cam_mat[1,2]

    u = fx * x + cx
    v = fy * y + cy

    points_2d = np.vstack([u,v]).T
    return points_2d


if success:
    print("running success")
    rvec2 = np.zeros((3,1), dtype=np.float32)
    tvec2 = np.zeros((3,1), dtype=np.float32)
    tvec3 = np.array([[0.0],
                 [0.0],
                 [0.7]], dtype=np.float32)
    R = np.array([
        [1, 0,  0],
        [0, 0,  -1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Convert back to rvec
    rvec3, _ = cv.Rodrigues(R)
    # Define 3D axis points (length = 5 cm)
    axis_length = 0.05  # meters
    axis_points = np.float32([
        [0, 0, 0],                   # origin
        [axis_length, 0, 0],         # X axis (red)
        [0, axis_length, 0],         # Y axis (green)
        [0, 0, axis_length]          # Z axis (blue)
    ])

    # Project to image
    imgpts, _ = cv.projectPoints(axis_points, rvec, tvec, cam_mat, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(img_points[0])



        # Project 3D points to image plane

    vertices_cv = vertices_array.copy()
    vertices_cv[:, [1,2]] = vertices_cv[:, [2,1]]  # swap y and z
    vertices_cv[:,1] *= -1  # invert y
    #faces_array = faces_array.astype(np.float32)

    objtoimg, _ = cv.projectPoints(vertices_array, rvec, tvec, cam_mat, dist_coeffs)
    objtoimg = np.int32(objtoimg).reshape(-1, 2)
    # points_2d = project_points_numpy(vertices_array, rvec, tvec, cam_mat)
    # for pt in points_2d.astype(int):
    #     cv.circle(frame, tuple(pt), 5, (0,255,0), -1)


    # Draw faces
    for face in faces_array:
        pts = objtoimg[face]
        cv.polylines(frame, [pts], True, (0,255,255), 2)

    #custom
    # for face in faces_array:
    #     pts = points_2d[face].astype(int)
    #     cv.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=1)
    #     cv.fillPoly(frame, [pts], color=(0,128,0))  # optional fill

    


    # Draw axes on frame
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3) # X - red
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3) # Y - green
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3) # Z - blue
    cv.imshow('img', frame)

    print("img pts", imgpts)

    svt = SolveVectorTrimesh()
    svt.solve(rvec, tvec)

    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    print("failed")