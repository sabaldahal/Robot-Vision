import os
import json
import re
import numpy as np

class DataFormatter():
    def __init__(self, data):
        self.data = data
        self.classes_category_map, self.keypoints_category_map = self.category_mapping()
        for k, v in self.keypoints_category_map.items():
            print(f"Keypoints Category Mapping: {k} -> {v}")

    def category_mapping(self):
        sorted_bboxes = sorted(self.data.all_classes_collection, key=lambda x: x.name.lower())  # case-insensitive sort
        sorted_keypoints = sorted(self.data.all_keypoints_collection, key=lambda x: x.name.lower())  # case-insensitive sort        
        mapping_classes = {f.name: idx + 1 for idx, f in enumerate(sorted_bboxes)}
        mapping_keypoints = {f.name: idx + 1 for idx, f in enumerate(sorted_keypoints)}
        print()
        return mapping_classes, mapping_keypoints
    
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
    
    def get_name_from_value(self, mapping, value):
        return next((k for k, v in mapping.items() if v == value), None)

    def filter_objects(self, bboxes, keypoints, min_visible=4):
        keys_to_remove = []
        for k,v in keypoints.items():
            visible_count = sum(1 for kp in v if kp["occluded"] == False)
            if visible_count < min_visible:
                keys_to_remove.append(k)
            # 
            #this section is specifically wriiten for spacecraft case - keypoint label version 3
            #may or may not work with other objects and other keypoint label version for the spacecraft
            #
            # filter the classes based on the minimum width in pixels
            #------start------
            else:
                if abs(v[0]["x"] - v[1]["x"]) < 4: #specify the number of pixels
                    keys_to_remove.append(k)
        for k in keys_to_remove:
            del keypoints[k]
            index = self.keypoints_category_map.get(k)
            key_to_delete = self.get_name_from_value(self.classes_category_map, index)
            del bboxes[key_to_delete]
        return bboxes, keypoints
    
    def get_corresponding_bbox(self, bboxes, index):
        name = self.get_name_from_value(self.classes_category_map, index)
        bbox = bboxes.get(name)
        x, y, w, h = self.format_bounding_box_to_COCO(bbox)
        return [x, y, w, h]
    
    def get_bbox_area(self, bboxes, index):
        name = self.get_name_from_value(self.classes_category_map, index)
        bbox = bboxes.get(name)
        x, y, w, h = self.format_bounding_box_to_COCO(bbox)
        area = w * h
        return area
    

    def export_data_COCO(self, file, saveAfterIterations):
        coco_data = None
        superCategory = "SpaceCrafts"
        if os.path.exists(file):
            with open(file, "r") as f:
                coco_data = json.load(f)
            print(f"Json file opened: {file}")
        else:
            coco_data = {
                "info":
                {
                    "description": "Spacecraft dataset",
                    "url": "unspecified",
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
                        "name": superCategory,
                        "supercategory": "none"
                    },
                    *(
                        {
                            "id": self.keypoints_category_map[kp_collection.name],
                            "name": kp_collection.name,
                            "supercategory": superCategory,
                            "keypoints": sorted(
                                                    (item.name for item in kp_collection.objects),
                                                    key=lambda n: n.lower()
                                                ),
                            "skeleton": []
                        } for kp_collection in self.data.all_keypoints_collection
                    )
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

            bbox, keypoints = self.filter_objects(bbox, keypoints)
            
            coco_data["images"].append(
                {
                    "id": image_index,
                    "file_name": f"{image_index:06d}.png",
                    "width": self.data.resx,
                    "height": self.data.resy
                }
            )
            coco_data["annotations"].extend(
                [
                    {
                        "id": image_index,
                        "image_id": image_index,
                        "category_id": self.keypoints_category_map[name],
                        "bbox": self.get_corresponding_bbox(bbox, self.keypoints_category_map[name]),
                        "area": self.get_bbox_area(bbox, self.keypoints_category_map[name]),
                        "segmentation": [],
                        "iscrowd": 0,
                        "keypoints": self.format_keypoints_to_COCO(values)
                    }
                    for i, (name, values) in enumerate(keypoints.items())
                ]
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

    def export_transformation_matrix(self, dir, image_index, matrix):
        file = os.path.join(dir, f"{image_index:06d}.txt")
        np.savetxt(file, matrix, fmt="%.7f")
