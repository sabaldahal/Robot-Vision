import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
# import pyrealsense2 as rs
import sys
sys.path.append('./estimator/solver')

from rvec_analyzer import *


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


coords_file = "./estimator/model/coords/format_3/coords.json"
model_path = "./estimator/weights/format_3.1/best.pt"
mesh_file = "./estimator/model/test.obj"
obj_points = []
img_points = []
keypointsArr = []
with open (coords_file, "r") as f:
    keypointsArr = json.load(f)



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
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
# profile = pipeline.start(config)
# depth_sensor, color_sensor, *_ = profile.get_device().query_sensors()
# color_sensor.set_option(rs.option.enable_auto_exposure, 1)
# color_sensor.set_option(rs.option.backlight_compensation, 0)
# color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
# color_sensor.set_option(rs.option.auto_exposure_priority, 1)

a = 1

video_path = "/home/sabal/code/spacecraft blender/latest/robot vision/Robot-Vision/local/test_videos/realsense_color_1.avi"
cap = cv.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()


# ----------------------------------------------start------**************************************-----------------------------
# from scipy.spatial.transform import Rotation as R

# # -----------------------------
# # Configuration
# # -----------------------------
# ALPHA_ROT = 0.1     # rotation smoothing (0.05–0.2)
# ALPHA_TRANS = 0.2   # translation smoothing
# MAX_ROT_JUMP_DEG = 25.0

# # -----------------------------
# # Persistent state
# # -----------------------------
# prev_q = None
# tvec_prev = None
# ----------------------------------------------end-----------**************************************-----------------------------

while a == 1:
    img_points = []
    obj_points = []

    ret, color_image = cap.read()
    if not ret:
        print("failed to grab frame")
        break 
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()

    # color_image = np.asanyarray(color_frame.get_data())
    cache_color_image = color_image.copy()
    result = model(color_image)[0]
    keypoints = result.keypoints.xy.cpu().numpy()
    bboxes = result.boxes.xyxy.cpu().numpy()
    class_names = result.names
    classes_predicted = result.boxes.cls.cpu().numpy().astype(int)

    predicted_keypoints = {}

    #detected keypoints
    for kps, cls_id in zip(keypoints, classes_predicted):
        p_class_name = class_names[cls_id]
        arr = []
        for x, y in kps:
            #img_points.append([x,y])
            arr.append([x,y])
        predicted_keypoints[p_class_name] = arr

    #match and filter
    filtered_keypoints = {k: v for k, v in keypointsArr.items() if k in predicted_keypoints}
    for k,v in predicted_keypoints.items():
        for t in v:
            img_points.append(t)
        for c in filtered_keypoints[k]:
            obj_points.append(c['location'])

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
        # print metrics
        print("Translation Vector:", tvec.flatten())
        
        objtoimg, _ = cv.projectPoints(vertices_array, rvec, tvec, cam_mat, dist_coeffs)
        objtoimg = np.int32(objtoimg).reshape(-1, 2)
        for face in faces_array:
            pts = objtoimg[face]
            cv.polylines(color_image, [pts], True, (0,255,255), 2)
        cv.imshow("Object Cam", color_image) 
    else:
        #no drawings
        cv.imshow("Object Cam", color_image) 
# ----------------------------------------------start------**************************************-----------------------------
    # try:
    #     success, rvec, tvec = cv.solvePnP(
    #         obj_points,
    #         img_points,
    #         cam_mat,
    #         dist_coeffs,
    #         flags=cv.SOLVEPNP_ITERATIVE
    #     )
    # except cv.error:
    #     cv.imshow("Object Cam", color_image)
    #     continue

    # if not success:
    #     cv.imshow("Object Cam", color_image)
    #     continue

    # # --------------------------------------------
    # # Pose refinement (IMPORTANT)
    # # --------------------------------------------
    # cv.solvePnPRefineLM(
    #     obj_points,
    #     img_points,
    #     cam_mat,
    #     dist_coeffs,
    #     rvec,
    #     tvec
    # )

    # # --------------------------------------------
    # # Rotation smoothing (quaternion)
    # # --------------------------------------------
    # R_curr, _ = cv.Rodrigues(rvec)
    # q_curr = R.from_matrix(R_curr).as_quat()

    # if prev_q is None:
    #     q_filt = q_curr
    # else:
    #     # Same hemisphere
    #     if np.dot(prev_q, q_curr) < 0:
    #         q_curr = -q_curr

    #     # Reject large jumps
    #     rot_delta = (
    #         R.from_quat(prev_q).inv() * R.from_quat(q_curr)
    #     ).magnitude()
    #     rot_delta_deg = np.degrees(rot_delta)

    #     if rot_delta_deg > MAX_ROT_JUMP_DEG:
    #         q_filt = prev_q
    #     else:
    #         q_filt = (1 - ALPHA_ROT) * prev_q + ALPHA_ROT * q_curr
    #         q_filt /= np.linalg.norm(q_filt)

    # prev_q = q_filt

    # R_filt = R.from_quat(q_filt).as_matrix()
    # rvec_filt, _ = cv.Rodrigues(R_filt)

    # # --------------------------------------------
    # # Translation smoothing
    # # --------------------------------------------
    # if tvec_prev is None:
    #     tvec_filt = tvec
    # else:
    #     tvec_filt = (1 - ALPHA_TRANS) * tvec_prev + ALPHA_TRANS * tvec

    # tvec_prev = tvec_filt

    # # --------------------------------------------
    # # Visualization (UNCHANGED LOGIC)
    # # --------------------------------------------
    # objtoimg, _ = cv.projectPoints(
    #     vertices_array,
    #     rvec_filt,
    #     tvec_filt,
    #     cam_mat,
    #     dist_coeffs
    # )

    # objtoimg = np.int32(objtoimg).reshape(-1, 2)

    # for face in faces_array:
    #     pts = objtoimg[face]
    #     cv.polylines(color_image, [pts], True, (0, 255, 255), 2)

    # cv.imshow("Object Cam", color_image)

# ----------------------------------------------end------**************************************-----------------------------
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# cv.waitKey(0)
cv.destroyAllWindows()