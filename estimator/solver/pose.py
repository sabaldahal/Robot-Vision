# Test file to test few features of the model
# does the following:
# runs inference
# runs pnp algorithm to get the 3d coordinates of the object in camera frame
# draws mesh with the predicted orientation on the image
# calculates rotation and translation error

#update
#works with both single-class and multi-class objects
#single-class object : entire object is considered as one class while calculating the bounding box
#multi-class object : An object is divided into multiple classes depending on its features to calculate the bounding box for each feature

import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
from modules.utils import *
from modules.utils.threeD_mesh import *
from modules.visualize import pose_viz
import argparse

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-t', '--testdir', type=str, default='./local/test_dataset/version3', help='Path to the test dataset directory')
parser.add_argument('-i', '--image', type=str, default='000147.png', help='Path to the input image')


args = parser.parse_args()
model_version = 'format_3.3'
coords_version = 'format_3'

image_filename = args.image
test_dataset_dir = args.testdir
img_path = os.path.join(test_dataset_dir, f'images/{image_filename}')
matrix_file = os.path.join(test_dataset_dir, f'transformation_matrices/{os.path.splitext(image_filename)[0]}.txt')


coords_file = f"./estimator/model/coords/{coords_version}/coords.json"
model_path = f"./estimator/weights/{model_version}/best.pt"
mesh_file = "./estimator/model/test.obj"


frame = cv2.imread(img_path)
yoloinstance = yolo.YOLODetect(model_path)
pnpinstance = pnp.PoseSolver()
pnpinstance.initialize(coords_file, yoloinstance.get_class_names())

classes, kpts, kptsconf, bboxes, bboxesconf = yoloinstance.run_inference(frame)

vertices_array = load_obj_vertices(mesh_file)
faces_array = load_obj_faces(mesh_file)

if model_version.startswith('format_2'):
    success, rvec, tvec, object_points, image_points = pnpinstance.format_single_class_keypoints_and_solve_pose(kpts)
else:
    success, rvec, tvec, object_points, image_points = pnpinstance.format_multi_class_keypoints_and_solve_pose(kpts, classes)

if success:

    Rvec_Blender_to_OpenCV, Tvec_Blender_to_OpenCV = load_blender_matrix(matrix_file)

    #analysis
    Rotation_Matrix, _ = cv2.Rodrigues(rvec)
    analyzer = error_analyzer.Analyzer()
    rotationalError = analyzer.getRotationError(Rotation_Matrix, Rvec_Blender_to_OpenCV)
    translationError = analyzer.getTranslationError(tvec, Tvec_Blender_to_OpenCV)

    print("Rotational Error: ", rotationalError)
    print("Translation Error: ", translationError)


    pose_viz.draw_pose(frame, rvec, tvec, vertices_array, faces_array)

else:
    print("failed")