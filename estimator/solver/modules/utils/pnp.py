import cv2
from .constants import *
import numpy as np
import json


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
    def solvepose(cls, object_points, image_points, rvec=None, tvec=None, use_Extrinsic_Guess=False):
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
    def format_multi_class_keypoints_and_solve_pose(cls, keypoints_predicted, classes_predicted, rvec=None, tvec=None, use_Extrinsic_Guess=False):
        obj_points, img_points = cls.format_multi_class_keypoints(keypoints_predicted, classes_predicted)
        s,r,t =  cls.solvepose(obj_points, img_points, rvec, tvec, use_Extrinsic_Guess)
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
    def format_single_class_keypoints_and_solve_pose(cls, keypoints_predicted, rvec=None, tvec=None, use_Extrinsic_Guess=False):
        obj_points, img_points = cls.format_single_class_keypoints(keypoints_predicted)
        s,r,t =  cls.solvepose(obj_points, img_points, rvec, tvec, use_Extrinsic_Guess)
        return (s, r, t, obj_points, img_points)