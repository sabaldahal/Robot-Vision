import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
# import pyrealsense2 as rs
import sys
from modules.utils import *
from modules.utils.threeD_mesh import *
from modules.visualize import pose_viz



model_version = 'format_3.4'
coords_version = 'format_3'

coords_file = f"./estimator/model/coords/{coords_version}/coords.json"
model_path = f"./estimator/weights/{model_version}/best.pt"
mesh_file = "./estimator/model/test.obj"


yoloinstance = yolo.YOLODetect(model_path)
pnpinstance = pnp.PoseSolver()
pnpinstance.initialize(coords_file, yoloinstance.get_class_names())

vertices_array = load_obj_vertices(mesh_file)
faces_array = load_obj_faces(mesh_file)



#video_path = "/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/animated video/spacecraft_animation0001-1800.mp4"
video_path = "/home/sabal/code/spacecraft blender/latest/robot vision/Robot-Vision/local/test_videos/realsense_color_1.avi"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

import datetime

paused = False
while cap.isOpened():
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'): # Press 'q' to exit
        break
    elif key == ord('p'): # Press 'p' to pause/resume
        paused = not paused

    if paused:
        continue

    ret, color_image = cap.read()
    if not ret:
        print("failed to grab frame")
        break 
    current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    seconds = int(time_ms / 1000)
    # timestamp = time_ms
    timestamp = str(datetime.timedelta(seconds=seconds))
    cv2.putText(color_image, f"Time: {timestamp}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(color_image, f"Frame: {current_frame_number}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    classes, kpts, kptsconf, bboxes, bboxesconf = yoloinstance.run_inference(color_image)

    if model_version.startswith('format_2'):
        success, rvec, tvec, object_points, image_points = pnpinstance.format_single_class_keypoints_and_solve_pose(kpts)
    else:
        success, rvec, tvec, object_points, image_points = pnpinstance.format_multi_class_keypoints_and_solve_pose(kpts, classes)

    window_name = "Pose Visualization"
    if success: 
        pose_viz.draw_pose(color_image, rvec, tvec, vertices_array, faces_array, wait=False, window=window_name)
    else:
        cv2.imshow(window_name, color_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()