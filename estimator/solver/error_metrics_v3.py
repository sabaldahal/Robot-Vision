

import pandas as pd
import numpy as np
import os
from datetime import date
from modules.utils import *
import cv2



ALL_MODELS_VERSION = ['format_3.1', 'format_3.2', 'format_3.3', 'format_2.1', 'format_2.2']
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

def main():
    RESULTS = []
    for m in ALL_MODELS_VERSION:
        test_data_dict = get_inference_results(m)
        RESULTS.extend(test_data_dict)
    df = pd.DataFrame(RESULTS)
    df.to_csv(outputfile, index=False)

def get_inference_results(modelversion):
    results = []
    model_path = f"./estimator/weights/{modelversion}/best.pt"
    coordsversion = modelversion.split('.', 1)[0]
    coords_file = f"./estimator/model/coords/{coordsversion}/coords.json"  

    yoloinstance = yolo.YOLODetect(model_path)
    pnpinstance = pnp.PoseSolver()
    pnpinstance.initialize(coords_file, yoloinstance.get_class_names())

    IS_MULTICLASS = modelversion.startswith('format_3')
    

    for img_name in sorted(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(images_dir, img_name)
        matrix_file = os.path.join(trans_mat_dir, f'{os.path.splitext(img_name)[0]}.txt')

        frame = cv2.imread(img_path)
        if frame is None:
            continue  

        classes, kpts, kptsconf, bboxes, bboxesconf = yoloinstance.run_inference(frame)

        #change this
        #depending on single class or multi class
        if IS_MULTICLASS:
            success, rvec, tvec, object_points, image_points = pnpinstance.format_multi_class_keypoints_and_solve_pose(kpts, classes)
        else:
            success, rvec, tvec, object_points, image_points = pnpinstance.format_single_class_keypoints_and_solve_pose(kpts)


        if not success:
            continue

        if tvec[2][0] < 0:
            rmatrix = np.array([
                (-1, 0, 0),
                (0, -1, 0),
                (0,  0, 1)
            ])

            rotation_matrix_1, _ = cv2.Rodrigues(rvec)
            modified_rotation_matrix = rmatrix @ rotation_matrix_1
            modified_rvec, _ = cv2.Rodrigues(modified_rotation_matrix)
            modified_tvec = -tvec

            success, rvec, tvec = pnpinstance.solvepose(object_points, image_points, modified_rvec, modified_tvec, use_Extrinsic_Guess=True)


        if success:
            ###Compare Matrices From Blender to OpenCV---------------------------------------       
            Transformation_Matrix_Blender_to_OpenCV = np.diag([1.0, -1.0, -1.0])
            matrix_from_file = np.loadtxt(matrix_file)
            R_Matrix_Blender = matrix_from_file[:3, :3]
            Tvec_Blender = matrix_from_file[:3, 3].reshape(3,1)
            Rvec_Blender_to_OpenCV = Transformation_Matrix_Blender_to_OpenCV @ R_Matrix_Blender
            Tvec_Blender_to_OpenCV = (Transformation_Matrix_Blender_to_OpenCV @ Tvec_Blender).reshape(3,1)

            #analysis
            Rotation_Matrix, _ = cv2.Rodrigues(rvec)
            analyzer = error_analyzer.Analyzer()
            rotationalError = analyzer.getRotationError(Rotation_Matrix, Rvec_Blender_to_OpenCV)
            translationError = analyzer.getTranslationError(tvec, Tvec_Blender_to_OpenCV)

            temp_dict = {
                    "Image": img_name,
                    "Rotational_Error_deg": rotationalError,
                    "Translational_Error_m": translationError,
                    "Model_Version": modelversion
            }

        confidence_dict = []
        if DEBUG:
            for kps, cls_id, b_conf, k_conf in zip(kpts, classes, bboxesconf, kptsconf):
                p_class_name = classes[cls_id]
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