import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
from modules.utils import yolo
from modules.visualize import bbox_kpts_viz
import argparse
import json

parser = argparse.ArgumentParser(description="Run Pose Estimation")
parser.add_argument('-m', '--modelformat', type=str, default='format_3.4', help='Path to the folder containing model weights')
parser.add_argument('-t', '--testdir', type=str, default='./local/test_dataset/version3', help='Path to the test dataset directory')
parser.add_argument('-i', '--image', type=str, default='000350.png', help='Path to the input image')


args = parser.parse_args()
model_version = args.modelformat

image_filename = args.image
test_dataset_dir = args.testdir
img_path = os.path.join(test_dataset_dir, f'images/{image_filename}')
model_path = f"./estimator/weights/{model_version}/best.pt"

def loadGroundTruthFromCOCOFile(filepath):
    data = {}
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def getNextDetails(data):
    kpts_arr = []
    bboxes = []
    for i in data["images"]:
        bboxes = [ x["bbox"] for x in data["annotations"] if x["image_id"] == i["id"]]
        
        for k in data["annotations"]:
            if k["image_id"] == i["id"]:
                each_class_kpts = []
                single_xy_kpt = []
                index = 1
                for eachkeypoint in k["keypoints"]:
                    if index % 3 == 0:
                        each_class_kpts.append(single_xy_kpt)
                        single_xy_kpt = []
                        index += 1
                        continue
                    single_xy_kpt.append(eachkeypoint)
                    
                    
        yield ()
    
    
    

frame = cv2.imread(img_path)

bboxes, kpts, classes, classes_name = loadGroundTruthFromCOCOFile(file)
bbox_kpts_viz.draw_bbox_keypoints(frame, bboxes, kpts, classes, classes_name)