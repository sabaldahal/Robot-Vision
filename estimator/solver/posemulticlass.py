# Test file to test few features of the model
# does the following:
# runs inference
# runs pnp algorithm to get the 3d coordinates of the object in camera frame
# draws mesh with the predicted orientation on the image
# calculates rotation and translation error

import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
from modules.utils import *
from modules.utils.threeD_mesh import *
import argparse

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-t', '--testdir', type=str, default='./local/from ubuntu/test_dataset/version3', help='Path to the test dataset directory')
parser.add_argument('-i', '--image', type=str, default='000394.png', help='Path to the input image')


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


success, rvec, tvec, object_points, image_points = pnpinstance.format_multi_class_keypoints_and_solve_pose(kpts, classes)
print(kpts)

if not success:
    print("Pose estimation failed")
    exit()

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
    axis_length = 0.05  # meters
    axis_points = np.float32([
        [0, 0, 0],                   # origin
        [axis_length, 0, 0],         # X axis (red)
        [0, axis_length, 0],         # Y axis (green)
        [0, 0, axis_length]          # Z axis (blue)
    ])

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

    print("Rotational Error: ", rotationalError)
    print("Translation Error: ", translationError)




    # Project to image
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, constants.Constants.cam_mat, constants.Constants.dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # Project 3D points to image plane
    vertices_cv = vertices_array.copy()
    vertices_cv[:, [1,2]] = vertices_cv[:, [2,1]]  # swap y and z
    vertices_cv[:,1] *= -1  # invert y

    ##original----------------------------
    objtoimg, _ = cv2.projectPoints(vertices_array, rvec, tvec, constants.Constants.cam_mat, constants.Constants.dist_coeffs)
    objtoimg = np.int32(objtoimg).reshape(-1, 2)

    ###keep this block commented
    # points_2d = project_points_numpy(vertices_array, rvec, tvec, cam_mat)
    # for pt in points_2d.astype(int):
    #     cv.circle(frame, tuple(pt), 5, (0,255,0), -1)

    #Draw faces
    for face in faces_array:
        pts = objtoimg[face]
        cv2.polylines(frame, [pts], True, (0,255,255), 2)

    # Draw axes on frame
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3) # X - red
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3) # Y - green
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3) # Z - blue
    cv2.imshow('img', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("failed")