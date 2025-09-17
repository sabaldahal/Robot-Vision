import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
import pyrealsense2 as rs
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


coords_file = "./estimator/model/coords.json"
model_path = "./estimator/weights/best.pt"
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

# Configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor, color_sensor, *_ = profile.get_device().query_sensors()
color_sensor.set_option(rs.option.enable_auto_exposure, 1)
color_sensor.set_option(rs.option.backlight_compensation, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
color_sensor.set_option(rs.option.auto_exposure_priority, 1)

a = 1

while a == 1:
    img_points = []
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())
    cache_color_image = color_image
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
        continue
    

    if success: 
        objtoimg, _ = cv.projectPoints(vertices_array, rvec, tvec, cam_mat, dist_coeffs)
        objtoimg = np.int32(objtoimg).reshape(-1, 2)
        for face in faces_array:
            pts = objtoimg[face]
            cv.polylines(color_image, [pts], True, (0,255,255), 2)
        cv.imshow("Object Cam", color_image) 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# cv.waitKey(0)
cv.destroyAllWindows()