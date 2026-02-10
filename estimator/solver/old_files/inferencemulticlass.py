from ultralytics import YOLO
import cv2
import os

model_version = 'format_3.3'
model_path = f"./estimator/weights/{model_version}/best.pt"
test_dataset_dir = f'local/test_dataset/version3'
img_path = os.path.join(test_dataset_dir, f'images/000143.png')
img = cv2.imread(img_path)

model = YOLO(model_path)

result = model(img_path)
print('printing')


keypoints = result[0].keypoints.xy.cpu().numpy()
bboxes = result[0].boxes.xyxy.cpu().numpy()
names = result[0].names
classes = result[0].boxes.cls.cpu().numpy().astype(int)
print('kps', keypoints)
print('boxes', bboxes)

class_colors = {
    0: (0, 255, 0),      # green
    1: (255, 0, 0),      # blue
    2: (0, 0, 255),      # red
    3: (0, 255, 255),    # yellow
    4: (255, 0, 255),    # purple
}

for kps, box, cls_id in zip(keypoints, bboxes, classes):
    idx = 0
    color = class_colors.get(cls_id, (255, 255, 255))
    class_name = names[cls_id]
    print('class id', class_name)
    x1, y1, a, b = map(int, box)
    cv2.rectangle(img, (x1,y1), (a,b), color, 1)
    cv2.putText(img, class_name, (x1, y1- 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    for x, y in kps:
        idx += 1
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
