import os
import json

file = '/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/working model/update_dec_4_2025/test_renders/images/_annotations.coco.json'


with open(file, "r") as f:
    coco_data = json.load(f)
    print(f"Json file opened: {file}")


temp_annotations = coco_data["annotations"]

index = 1
final_annotations = []
for annotation in temp_annotations:
    for inner in annotation:
        inner['id'] = index
        final_annotations.append(inner)
        index += 1

coco_data["annotations"] = final_annotations

newfile = '/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/working model/update_dec_4_2025/test_renders/images/_annotations_corrected.coco.json'
with open(newfile, "w") as f:
    json.dump(coco_data, f, indent=4)
print(f"[AUTO SAVE] saved file {newfile}")