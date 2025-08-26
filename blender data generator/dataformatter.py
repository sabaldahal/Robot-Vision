import os
import json
import re

class DataFormatter():
    def __init__(self, data):
        self.data = data
    
    def clip_bounding_box(self, bbox):
        x, y, a, b = bbox
        x = max(0, min(self.data.resx, x))
        a = max(0, min(self.data.resx, a))
        y = max(0, min(self.data.resy, y))
        b = max(0, min(self.data.resy, b))      
        return (x, y, a, b)

    def format_bounding_box_to_YOLO(self, bbox):
        bbox = self.clip_bounding_box(bbox)
        x, y, a, b = bbox
        xcenter = ((x + a)/2)/self.data.resx
        ycenter = ((y+b)/2)/self.data.resy
        width = (a-x)/self.data.resx
        height = (b-y)/self.data.resy
        return xcenter, ycenter, width, height

    def format_bounding_box_to_COCO(self, bbox):
        bbox = self.clip_bounding_box(bbox)
        x, y, a, b = bbox
        width = a-x
        height = b-y
        return x, y, width, height
    
    def clip_keypoints(self, keypoint):
        x, y = keypoint
        x = max(0, min(self.data.resx, x))
        y = max(0, min(self.data.resy, y))
        return (x, y)
    
    def format_keypoints_to_COCO(self, keypoints):
        keypoints_coco = []
        for k in keypoints:
            x, y = self.clip_keypoints((k["x"], k["y"]))
            v = 2
            if k["occluded"]:
                v = 1
            if not k["inFrame"]:
                v = 0
            keypoints_coco.extend([x,y,v])
        return keypoints_coco

    def format_keypoints_to_YOLO(self, keypoints):
        keypoints_yolo = []
        for k in keypoints:
            x, y = self.clip_keypoints((k["x"], k["y"]))
            x = x/self.data.resx
            y = y/self.data.resy
            v = 2
            if k["occluded"]:
                v = 1
            if not k["inFrame"]:
                v = 0
            keypoints_yolo.append((x,y,v))
        return keypoints_yolo
    
    def export_data_COCO(self, file, saveAfterIterations):
        coco_data = None
        if os.path.exists(file):
            with open(file, "r") as f:
                coco_data = json.load(f)
            print(f"Json file opened: {file}")
        else:
            keypoints_name_ascending = sorted(
                (item.name for item in self.data.keypoint_collection.objects),
                key=lambda n: (
                    re.sub(r'_\d+$', '', n),                 # prefix
                    int(re.search(r'(\d+)$', n).group())     # numeric part
                )
            )
            coco_data = {
                "info":
                {
                    "description": "Spacecraft dataset",
                    "url": "www.google.com",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "Sabal Dahal",
                    "date_created": "2025/08/07"
                },
                "licenses": 
                {
                    "id": 1,
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "name": "CC BY 4.0"
                },
                "categories": 
                [
                    {
                        "id": 0,
                        "name": "SpaceCrafts",
                        "supercategory": "none"
                    },
                    {
                        "id": 1,
                        "name": "spacecraft",
                        "supercategory": "SpaceCrafts",
                        "keypoints": keypoints_name_ascending,
                        "skeleton": []
                    }
                ],
                "images": [],
                "annotations": []
            }
        
        totalSaved = 0
        while True:
            data = yield
            if isinstance(data, bool) and data:
                with open(file, "w") as f:
                    json.dump(coco_data, f, indent=4)
                print(f"[FINAL SAVE] All annotations saved to {file}")
                return
            
            image_index, bbox, keypoints = data
            x, y, w, h = self.format_bounding_box_to_COCO(bbox)
            area = w * h
            keypoints_coco = self.format_keypoints_to_COCO(keypoints)
            coco_data["images"].append(
                {
                    "id": image_index,
                    "file_name": f"{image_index:06d}.png",
                    "width": self.data.resx,
                    "height": self.data.resy
                }
            )
            coco_data["annotations"].append(
                {
                    "id": image_index,
                    "image_id": image_index,
                    "category_id": 1,
                    "bbox": [
                        x,
                        y,
                        w,
                        h
                    ],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                    "keypoints": keypoints_coco,
                }
            )
            totalSaved += 1
            if totalSaved >= saveAfterIterations:
                with open(file, "w") as f:
                    json.dump(coco_data, f, indent=4)
                print(f"[AUTO SAVE] saved file {file}")
                totalSaved = 0



            
    def export_data_YOLO(self, label_dir, image_index, bbox, keypoints):
        xcenter, ycenter, width, height = self.format_bounding_box_to_YOLO(bbox)
        keypoints_yolo = self.format_keypoints_to_YOLO(keypoints)
        yolo_line = f"0 {xcenter} {ycenter} {width} {height}"
        for k in keypoints_yolo:
            x, y, v = k
            yolo_line = yolo_line + f" {x} {y} {v}"        
        label_path = os.path.join(label_dir, f"{image_index:06d}.txt")
        with open(label_path, "w") as f:
            f.write(yolo_line + "\n")


