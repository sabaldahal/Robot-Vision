import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np


if not hasattr(np, "infty"):
    np.infty = np.inf

import trimesh
import argparse

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-i', '--image', type=str, default='2.png', help='Path to the input image')


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


mesh = trimesh.load("./estimator/model/spacecraft.obj", force='mesh')


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

vertices_array = load_obj_vertices("./estimator/model/spacecraft.obj")
faces_array = load_obj_faces("./estimator/model/spacecraft.obj")


success, rvec, tvec = cv.solvePnP(
    obj_points, 
    img_points, 
    cam_mat, 
    dist_coeffs
)






def to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    print(x)
    print(y)
    print(z)
    return np.array([x, y, z]) 


if success:
    print("running success")
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
    imgpts, _ = cv.projectPoints(vertices_array, rvec, tvec, cam_mat, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw faces
    for face in faces_array:
        pts = imgpts[face]
        cv.polylines(frame, [pts], True, (0,255,0), 2)
    

    


    # Draw axes on frame
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3) # X - red
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3) # Y - green
    cv.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3) # Z - blue
    cv.imshow('img', frame)


    #trimesh
    # Invert pose to go from camera → world
    Ro, _ = cv.Rodrigues(rvec)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = Ro
    tt = tvec.flatten()
    pose_matrix[:3, 3] = tt * 0.2
    pose_matrix_inverse = np.linalg.inv(pose_matrix)
    axis_correction = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
    ])
    Mtot = axis_correction @ Ro
    T= np.eye(4)
    T[:3, :3] = Mtot
    axis_conv = np.array([
        [1,  0, 0, 0],  # X stays
        [0, 0, 1, 0],  # Y flipped
        [0,  1, 0, 0],  # Z stays
        [0,  0, 0, 1]
        ], dtype=float)

    print(rvec)
    a = to_euler(Ro)
    Tz = trimesh.transformations.rotation_matrix(
        angle=a[2],
        direction=[0, 0, 1],  # Z-axis
        point=[0, 0, 0]       # rotate around origin
    )
    angle_rad  = np.linalg.norm(rvec)
    angle_deg = np.degrees(angle_rad)

    

    scene = trimesh.Scene(mesh)
    # scene.camera.resolution = [1280, 720]
    # scene.camera.fov = (2 * np.degrees(np.arctan(720/(2*fy))),  # vertical FOV
    #                 2 * np.degrees(np.arctan(1280/(2*fx)))) 
    R_cv, _ = cv.Rodrigues(rvec)

# --- Step 2: Convert OpenCV coords -> Trimesh coords ---
# OpenCV: X right, Y down, Z forward
# Trimesh: X right, Y forward, Z up
    axis_convert = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0,1, 0]
    ])

    axis_conv2 = np.array([
    [-1,  0,  0, 0],  # flip X
    [ 0,  0,  1, 0],  # Y = Z
    [ 0, -1,  0, 0],  # Z = -Y
    [ 0,  0,  0, 1]
    ])
    
    R_tm = axis_convert @ R_cv @ axis_convert.T

    n_tm = axis_conv2 @ pose_matrix
    # --- Step 3: Build 4x4 camera transform (world → camera) ---
    T = np.eye(4)
    T[:3, :3] = R_tm


    camera_tf = np.linalg.inv(T)

    # --- Step 5: Assign camera transform ---
    scene.camera_transform = pose_matrix
    scene.show()



    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    print("failed")