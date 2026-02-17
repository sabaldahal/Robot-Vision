import cv2
from .constants import *
import numpy as np
import json
import time


class PoseSolver:

    keypoints_3d = None
    class_names = None
    prev_rvec = None
    prev_tvec = None

    @classmethod
    def initialize(cls, coordsfile, class_names):
        with open(coordsfile, "r") as f:
            cls.keypoints_3d = json.load(f)
        cls.class_names = class_names

    @classmethod
    def solvepose(cls, object_points, image_points, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front = True):
        if image_points is None or image_points.size == 0:
            print("pnp::solvepose: No image points provided, cannot solve pose.")
            return (False, None, None)
        start_time_all = time.perf_counter()
        start_time_cpu = time.process_time()
        try:
            if use_Extrinsic_Guess:
                success, result_rvec, result_tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    Constants.cam_mat,
                    Constants.dist_coeffs,
                    rvec,
                    tvec,
                    useExtrinsicGuess=use_Extrinsic_Guess
                )
            else:
                success, result_rvec, result_tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    Constants.cam_mat,
                    Constants.dist_coeffs
                )
        except Exception as ex:
            success = False
            print("pnp::solvepose:", ex)
            return (success, None, None)


        else:
            mid_time_all = time.perf_counter()
            mid_time_cpu = time.process_time()
            if success and bring_object_to_front:
                if result_tvec[2][0] < 0:
                    rmatrix = np.array([
                        (-1, 0, 0),
                        (0, -1, 0),
                        (0,  0, 1)
                    ])

                    rotation_matrix_1, _ = cv2.Rodrigues(result_rvec)
                    modified_rotation_matrix = rmatrix @ rotation_matrix_1
                    modified_rvec, _ = cv2.Rodrigues(modified_rotation_matrix)
                    modified_tvec = -result_tvec

                    try:
                        success, result_rvec, result_tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            Constants.cam_mat,
                            Constants.dist_coeffs,
                            modified_rvec,
                            modified_tvec,
                            useExtrinsicGuess = True
                        )
                    except Exception as ex1:
                        success = False
                        print("pnp::solvepose: Failed to Solve Pose After transforming mirrored pose and Running PnP")
                        print("pnp::solvepose", ex1)

            end_time_all = time.perf_counter()
            end_time_cpu = time.process_time()
            all_actual = (end_time_all - start_time_all) * 1000
            first_actual = (mid_time_all - start_time_all)* 1000
            second_actual = (end_time_all - mid_time_all)* 1000
            all_cpu = (end_time_cpu - start_time_cpu)* 1000
            first_cpu = (mid_time_cpu - start_time_cpu)* 1000
            second_cpu = (end_time_cpu - mid_time_cpu)* 1000
            print(f'pnp:solvepose Speed Actual: {all_actual}, First: {first_actual}, Second: {second_actual}')
            print(f'pnp:solvepose Speed CPU: {all_cpu}, First: {first_cpu}, Second: {second_cpu}')
                        
            return (success, result_rvec, result_tvec)
        
    @classmethod
    def format_multi_class_keypoints(cls, keypoints_predicted, classes_predicted):
        img_points = []
        obj_points = []
        predicted_keypoints = {}
        for kps, cls_id in zip(keypoints_predicted, classes_predicted):
            p_class_name = cls.class_names[cls_id]
            arr = []
            for x, y in kps:
                arr.append([x,y])
            predicted_keypoints[p_class_name] = arr

        #match and filter
        filtered_keypoints = {k: v for k, v in cls.keypoints_3d.items() if k in predicted_keypoints}
        for k,v in predicted_keypoints.items():
            for t in v:
                img_points.append(t)
            for c in filtered_keypoints[k]:
                obj_points.append(c['location'])

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points  = np.array(img_points,  dtype=np.float32)

        return (obj_points, img_points)

    @classmethod
    def format_multi_class_keypoints_and_solve_pose(cls, keypoints_predicted, classes_predicted, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front=True):
        obj_points, img_points = cls.format_multi_class_keypoints(keypoints_predicted, classes_predicted)
        s,r,t =  cls.solvepose(obj_points, img_points, rvec, tvec, use_Extrinsic_Guess, bring_object_to_front)
        return (s,r,t, obj_points, img_points)
        
    @classmethod
    def format_single_class_keypoints(cls, keypoints_predicted):
        img_points = []
        obj_points = []
        for k in cls.keypoints_3d:
            obj_points.append(k['location'])

        for kps in keypoints_predicted:
            for x, y in kps:
                img_points.append([x,y])

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points  = np.array(img_points,  dtype=np.float32)

        return (obj_points, img_points)
    
    @classmethod
    def format_single_class_keypoints_and_solve_pose(cls, keypoints_predicted, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front=True):
        obj_points, img_points = cls.format_single_class_keypoints(keypoints_predicted)
        s,r,t =  cls.solvepose(obj_points, img_points, rvec, tvec, use_Extrinsic_Guess, bring_object_to_front)
        return (s, r, t, obj_points, img_points)