import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
from modules.utils import yolo
from modules.visualize import bbox_kpts_viz
import argparse

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-t', '--testdir', type=str, default='./local/test_dataset/version3', help='Path to the test dataset directory')
parser.add_argument('-i', '--image', type=str, default='000394.png', help='Path to the input image')


args = parser.parse_args()
model_version = 'format_3.2'

image_filename = args.image
test_dataset_dir = args.testdir
img_path = os.path.join(test_dataset_dir, f'images/{image_filename}')
model_path = f"./estimator/weights/{model_version}/best.pt"

frame = cv2.imread(img_path)
yoloinstance = yolo.YOLODetect(model_path)
classes_name = yoloinstance.get_class_names()

classes, kpts, kptsconf, bboxes, bboxesconf = yoloinstance.run_inference(frame)

frame = bbox_kpts_viz.draw_bbox_keypoints(frame, bboxes, kpts, classes, classes_name, show_image=False, wait=False)
bbox_kpts_viz.draw_confidence_scores(frame, bboxesconf, kptsconf, classes, classes_name)