import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
import sys
sys.path.append('./estimator/solver')

from trimeshv import *


if not hasattr(np, "infty"):
    np.infty = np.inf

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



coords_file = "./estimator/model/coords/format_2.1/coords.json"
model_path = "./estimator/weights/format_2.2/best.pt"
mesh_file = "./estimator/model/test.obj"
obj_points = []
img_points = []
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


model = YOLO(model_path)

vertices_array = load_obj_vertices(mesh_file)
faces_array = load_obj_faces(mesh_file)



video_path = "/home/sabal/code/spacecraft blender/latest/robot vision/Robot-Vision/local/test_videos/realsense_color_1.avi"

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()


a = 1

while a == 1:
    img_points = []

    ret, color_image = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cache_color_image = color_image.copy()
    result = model(color_image)[0]
    keypoints = result.keypoints.xy.cpu().numpy()
    bboxes = result.boxes.xyxy.cpu().numpy()

    #detected keypoints
    for kps in keypoints:
        for x, y in kps:
            img_points.append([x,y])
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points  = np.array(img_points,  dtype=np.float32)
    try:
        success, rvec, tvec = cv.solvePnP(
            obj_points, 
            img_points, 
            cam_mat, 
            dist_coeffs
        )
    except:
        print("could not solve")
        #no drawings
        cv.imshow("Object Cam", color_image) 
        continue
    

    if success: 
        objtoimg, _ = cv.projectPoints(vertices_array, rvec, tvec, cam_mat, dist_coeffs)
        objtoimg = np.int32(objtoimg).reshape(-1, 2)
        for face in faces_array:
            pts = objtoimg[face]
            cv.polylines(color_image, [pts], True, (0,255,255), 2)
        cv.imshow("Object Cam", color_image) 
    else:
        #no drawings
        cv.imshow("Object Cam", color_image) 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# cv.waitKey(0)
cv.destroyAllWindows()