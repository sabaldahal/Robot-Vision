import os
import json

dir = "local/close up dataset/version1/images"

test_annotations_dir = "local/close up dataset/test/images"
train_annotations_dir = "local/close up dataset/train/images"

annotations_file_name = "_annotations.coco.json"

test_annotations_file_name = os.path.join(test_annotations_dir, annotations_file_name)
train_annotations_file_name = os.path.join(train_annotations_dir, annotations_file_name)
annotations_file_path = os.path.join(dir, annotations_file_name)

split_start_index = 2700

annotations_dict = {}
with open(annotations_file_path, 'r') as f:
    annotations_dict = json.load(f)


train_images = annotations_dict["images"][:split_start_index]
test_images = annotations_dict["images"][split_start_index:]


train_annotations = []
test_annotations = []

for k in annotations_dict["annotations"]:
    if k["image_id"] < split_start_index:
        train_annotations.append(k)
    else:
        test_annotations.append(k)


annotations_dict["images"] = train_images
annotations_dict["annotations"] = train_annotations

final_test_annotations = annotations_dict.copy()
final_test_annotations["images"] = test_images
final_test_annotations["annotations"] = test_annotations

with open (test_annotations_file_name, "w") as f:
    json.dump(final_test_annotations, f)

with open (train_annotations_file_name, "w") as f:
    json.dump(annotations_dict, f)