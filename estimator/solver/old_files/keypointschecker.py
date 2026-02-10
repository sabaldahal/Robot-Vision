from ultralytics import YOLO
import cv2
import os


# frames_to_extract = [210, 1651, 1581]
# get_image_at = 0

# img_path = f'local/test_videos/frame_analysis/frame_{frames_to_extract[get_image_at]}.jpg'
# print(img_path)


model_version = 'format_3.3'
model_path = f"./estimator/weights/{model_version}/best.pt"
test_dataset_dir = f'local/from ubuntu/test_dataset/version3'
img_path = os.path.join(test_dataset_dir, f'images/000394.png')

img = cv2.imread(img_path)

model = YOLO(model_path)

result = model(img_path, conf=0.6)
print('printing')


keypoints = result[0].keypoints.xy.cpu().numpy()
bboxes = result[0].boxes.xyxy.cpu().numpy()
names = result[0].names
classes = result[0].boxes.cls.cpu().numpy().astype(int)


#confidence values
boxes_conf = result[0].boxes.conf.cpu().numpy()
kps_conf = result[0].keypoints.conf.cpu().numpy()

print('original raw conf-----------------')
print(boxes_conf)
print()
print(kps_conf)

print('original end--------------')


class_colors = {
    0: (0, 255, 0),      # green
    1: (255, 0, 0),      # blue
    2: (0, 0, 255),      # red
    3: (0, 255, 255),    # yellow
    4: (255, 0, 255),    # purple
}

for kps, box, cls_id, b_conf, k_conf in zip(keypoints, bboxes, classes, boxes_conf, kps_conf):
    idx = 0
    color = class_colors.get(cls_id, (255, 255, 255))
    class_name = names[cls_id]
    print('class id ', class_name)
    print('class confidence: ', b_conf)
    print('k_conf: ', k_conf)

    x1, y1, a, b = map(int, box)
    cv2.rectangle(img, (x1,y1), (a,b), color, 1)
    cv2.putText(img, class_name, (x1, y1- 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for x, y in kps:
        idx += 1
        print(f'keypoint {idx}: conf = {k_conf[idx-1]}')
        cv2.circle(img, (int(x), int(y)), 1, color, -1)
        cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
