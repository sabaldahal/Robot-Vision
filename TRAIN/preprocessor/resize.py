import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==== CONFIG ====
IMG_DIR = "dataset/images"
YOLO_LABEL_DIR = "dataset/labels_yolo"
COCO_LABEL_PATH = "dataset/labels_coco.json"  # optional
OUT_DIR = "dataset_processed"
IMG_SIZE = (640, 640)
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test

os.makedirs(OUT_DIR, exist_ok=True)

def resize_yolo_keypoints_label(label_path, orig_w, orig_h, new_w, new_h):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    resized_lines = []
    for line in lines:
        parts = line.strip().split()
        cls_id = parts[0]
        nums = list(map(float, parts[1:]))

        # YOLO keypoints: class cx cy w h kpx1 kpy1 ... visibility
        # normalize by image size
        for i in range(len(nums)):
            if i % 2 == 0:  # x
                nums[i] *= (new_w / orig_w)
            else:  # y
                nums[i] *= (new_h / orig_h)
        resized_lines.append(f"{cls_id} " + " ".join(map(str, nums)) + "\n")

    return resized_lines

def process_dataset(splits):
    for split_name, split_files in splits.items():
        out_img_dir = os.path.join(OUT_DIR, split_name, "images")
        out_lbl_dir = os.path.join(OUT_DIR, split_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        print(f"Processing {split_name} ({len(split_files)} images)...")
        for img_name in tqdm(split_files):
            img_path = os.path.join(IMG_DIR, img_name)
            lbl_path = os.path.join(YOLO_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, IMG_SIZE)

            out_img_path = os.path.join(out_img_dir, img_name)
            cv2.imwrite(out_img_path, resized)

            if os.path.exists(lbl_path):
                new_lbl_lines = resize_yolo_keypoints_label(lbl_path, orig_w, orig_h, IMG_SIZE[0], IMG_SIZE[1])
                out_lbl_path = os.path.join(out_lbl_dir, os.path.basename(lbl_path))
                with open(out_lbl_path, 'w') as f:
                    f.writelines(new_lbl_lines)

    print("Done resizing and splitting YOLO data!")

    def split_coco_json(coco_path, splits):
        with open(coco_path, 'r') as f:
            coco = json.load(f)

        img_id_map = {img['file_name']: img['id'] for img in coco['images']}
        anns_by_img = {}
        for ann in coco['annotations']:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)

        for split_name, img_files in splits.items():
            split_json = {"images": [], "annotations": [], "categories": coco['categories']}
            for img_name in img_files:
                if img_name not in img_id_map:
                    continue
                img_id = img_id_map[img_name]
                split_json["images"].append(next(img for img in coco['images'] if img['id'] == img_id))
                split_json["annotations"].extend(anns_by_img.get(img_id, []))

            with open(os.path.join(OUT_DIR, f"coco_{split_name}.json"), 'w') as f:
                json.dump(split_json, f)

        print("COCO JSON split done.")


process_dataset()
