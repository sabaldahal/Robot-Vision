import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import json
import numpy as np
import pandas as pd

import sys
sys.path.append('./estimator/solver')
from rvec_analyzer import *

if not hasattr(np, "infty"):
    np.infty = np.inf


model_version = 'format_2.2'
coords_version = 'format_2.1'
test_dataset_version = 'format_2.1'
test_output_version = 1

test_dataset_dir = f'./estimator/test_dataset/{test_dataset_version}'
images_dir = os.path.join(test_dataset_dir, 'images')
trans_mat_dir = os.path.join(test_dataset_dir, 'transformation_matrices')
coords_file = f"./estimator/model/coords/{coords_version}/coords.json"
model_path = f"./estimator/weights/{model_version}/best.pt"

mesh_file = "./estimator/model/test.obj"

test_output_version_dir = f'{model_version}_{test_output_version}'
os.makedirs(f'./estimator/test results/{test_output_version_dir}', exist_ok=True)
outputfile = f'./estimator/test results/{test_output_version_dir}/{model_version}_{test_output_version}_pose_errors.csv'

obj_points = []

RESULTS = []

fx = 915.5166015625
fy = 915.607421875
cx = 629.287109375
cy = 356.802307128906

cam_mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

with open (coords_file, "r") as f:
    keypointsArr = json.load(f)
for k in keypointsArr:
    obj_points.append(k['location'])

model = YOLO(model_path)

for img_name in sorted(os.listdir(images_dir)):
    img_points = []
    keypointsArr = []

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(images_dir, img_name)
    matrix_file = os.path.join(trans_mat_dir, f'{os.path.splitext(img_name)[0]}.txt')

    frame = cv.imread(img_path)
    if frame is None:
        continue

    result = model(img_path)[0]
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
    except Exception as ex:
        success = False
        print("Could not solve pose")
        print(ex)


    if success:
        ###Compare Matrices From Blender to OpenCV---------------------------------------       
        S = np.diag([1.0, -1.0, -1.0])
        matrix_from_file = np.loadtxt(matrix_file)
        R_b = matrix_from_file[:3, :3]
        T_b = matrix_from_file[:3, 3].reshape(3,1)
        R_b2cv = S @ R_b
        T_b2cv = (S @ T_b).reshape(3,1)

        #analysis
        Ro, _ = cv.Rodrigues(rvec)
        analyzer = Analyzer()
        rotationalError = analyzer.getRotationError(Ro, R_b2cv)
        translationError = analyzer.getTranslationError(tvec, T_b2cv)

        RESULTS.append({
            "Image": img_name,
            "Rotational_Error_deg": rotationalError,
            "Translational_Error_m": translationError
        })
    else:
        print("failed")

df = pd.DataFrame(RESULTS)
df.to_csv(outputfile, index=False)
