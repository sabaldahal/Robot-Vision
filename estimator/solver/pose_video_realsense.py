import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
import pyrealsense2 as rs
from modules.utils import *
from modules.utils.threeD_mesh import *
from modules.visualize import pose_viz




model_version = 'format_3.3'
coords_version = 'format_3'

coords_file = f"./estimator/model/coords/{coords_version}/coords.json"
model_path = f"./estimator/weights/{model_version}/best.pt"
mesh_file = "./estimator/model/test.obj"



yoloinstance = yolo.YOLODetect(model_path)
pnpinstance = pnp.PoseSolver()
pnpinstance.initialize(coords_file, yoloinstance.get_class_names())

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



while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())
    cache_color_image = color_image.copy()
    classes, kpts, kptsconf, bboxes, bboxesconf = yoloinstance.run_inference(color_image)

    if model_version.startswith('format_2'):
        success, rvec, tvec, object_points, image_points = pnpinstance.format_single_class_keypoints_and_solve_pose(kpts)
    else:
        success, rvec, tvec, object_points, image_points = pnpinstance.format_multi_class_keypoints_and_solve_pose(kpts, classes)


    if success: 
        pose_viz.draw_pose(color_image, rvec, tvec, vertices_array, faces_array, wait=False)
    else:
        cv2.imshow("Object Cam", color_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv.waitKey(0)
cv2.destroyAllWindows()