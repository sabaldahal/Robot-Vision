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


# Configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline_profile = pipeline.start(config)
# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

a = 1
while a == 1:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    result = model(color_image)[0]
    keypoints = result.keypoints.xy.cpu().numpy()
    bboxes = result.boxes.xyxy.cpu().numpy()

    #detected keypoints
    for kps in keypoints:
        for x, y in kps:
            img_points.append([x,y])
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points  = np.array(img_points,  dtype=np.float32)
    success, rvec, tvec = cv.solvePnP(
        obj_points, 
        img_points, 
        cam_mat, 
        dist_coeffs
    )

    if success:  
        svt = SolveVectorTrimesh()
        svt.solve(rvec, tvec)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("failed")

    a = 3