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
from datetime import date

ALL_MODELS_VERSION = ['format_3', 'format_3.1', 'format_3.2', 'format_3.3']
TEST_DATASET_VERSION = 'version3'
TEST_OUTPUT_VERSION = 1
DEFAULT_OUTPUT_FOLDER = 'version_2'
OUTPUT_FOLDER = 'debug_confidence_1'
USE_DEFAULT = False
DEBUG = True

if USE_DEFAULT:
    OUTPUT_FOLDER = DEFAULT_OUTPUT_FOLDER

test_dataset_dir = f'./estimator/test_dataset/{TEST_DATASET_VERSION}'
images_dir = os.path.join(test_dataset_dir, 'images')
trans_mat_dir = os.path.join(test_dataset_dir, 'transformation_matrices')
mesh_file = "./estimator/model/test.obj"
test_output_version_dir = f'./estimator/test results/{OUTPUT_FOLDER}/v{TEST_OUTPUT_VERSION}_{date.today()}'
os.makedirs(test_output_version_dir, exist_ok=True)
outputfile = f'{test_output_version_dir}/file_multiclass_{TEST_OUTPUT_VERSION}_{date.today()}_pose_errors.csv'



fx = 915.5166015625
fy = 915.607421875
cx = 629.287109375
cy = 356.802307128906

cam_mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)



def main():
    RESULTS = []
    for m in ALL_MODELS_VERSION:
        test_data_dict = get_inference_results(m)
        RESULTS.extend(test_data_dict)
        

        

    df = pd.DataFrame(RESULTS)
    df.to_csv(outputfile, index=False)

def get_inference_results(modelversion):

    obj_points = []
    img_points = []
    results = []
    keypointsArr = {}
    model_path = f"./estimator/weights/{modelversion}/best.pt"
    coordsversion = modelversion.split('.', 1)[0]
    coords_file = f"./estimator/model/coords/{coordsversion}/coords.json"  
    model = YOLO(model_path)
    with open (coords_file, "r") as f:
        keypointsArr = json.load(f)
        


    for img_name in sorted(os.listdir(images_dir)):
        img_points = []
        obj_points = []

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(images_dir, img_name)
        matrix_file = os.path.join(trans_mat_dir, f'{os.path.splitext(img_name)[0]}.txt')

        frame = cv.imread(img_path)
        if frame is None:
            continue

        result = model(img_path, conf=0.6)[0]
        keypoints = result.keypoints.xy.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()
        class_names = result.names
        classes_predicted = result.boxes.cls.cpu().numpy().astype(int)
        #confidence values
        boxes_conf = result.boxes.conf.cpu().numpy()
        kps_conf = result.keypoints.conf.cpu().numpy()

        predicted_keypoints = {}

    #detected keypoints
        for kps, cls_id in zip(keypoints, classes_predicted):
            p_class_name = class_names[cls_id]
            arr = []
            for x, y in kps:
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

            temp_dict = {
                    "Image": img_name,
                    "Rotational_Error_deg": rotationalError,
                    "Translational_Error_m": translationError,
                    "Model_Version": modelversion
            }

        confidence_dict = []
        if DEBUG:
            for kps, cls_id, b_conf, k_conf in zip(keypoints, classes_predicted, boxes_conf, kps_conf):
                p_class_name = class_names[cls_id]
                index = 0
                confidence_dict.append({
                    'Image': img_name,
                    'Rotational_Error_deg': rotationalError,
                    'Translational_Error_m': translationError,
                    'Model_Version': modelversion,
                    'Class_Name' : f'{p_class_name}',
                    'Visibility' : 'visible',
                    'Bbox_conf' : b_conf
                })

                for x, y in kps:
                    confidence_dict.append({
                        'Image': img_name,
                        'Rotational_Error_deg': rotationalError,
                        'Translational_Error_m': translationError,
                        'Model_Version': modelversion,
                        'Class_Name' : f'{p_class_name}',
                        'Visibility' : 'visible',
                        'Keypoint_Name' : f'{p_class_name}_kpt_{index}',
                        'keypoint_conf' : k_conf[index]
                    })
                    index += 1 

            
            results.extend(confidence_dict)

        else:
            results.append(temp_dict)

    return results




main()