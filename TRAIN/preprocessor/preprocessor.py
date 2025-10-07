import os
import json
import sys 

sys.path.append('./TRAIN/preprocessor')
from resize import *

# ==== CONFIG ====
IMG_DIR = "dataset/images"
YOLO_LABEL_DIR = "dataset/labels_yolo"
COCO_LABEL_PATH = "dataset/labels_coco.json"
OUT_DIR = "dataset_processed"
IMG_SIZE = (640, 640)
SPLIT_RATIO = (0.7, 0.2, 0.1) 

os.makedirs(OUT_DIR, exist_ok=True)

# split dataset
images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))])
train_imgs, temp_imgs = train_test_split(images, test_size=1 - SPLIT_RATIO[0], random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=SPLIT_RATIO[2]/(SPLIT_RATIO[1]+SPLIT_RATIO[2]), random_state=42)
splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

