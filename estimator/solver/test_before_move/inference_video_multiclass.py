from ultralytics import YOLO
import cv2
import os

model_version = 'format_3.1'
model_path = f"./estimator/weights/{model_version}/best.pt"
video_path = "local/test.mov"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()
while True:

    ret, img = cap.read()

    model = YOLO(model_path)

    result = model(img)


    keypoints = result[0].keypoints.xy.cpu().numpy()
    bboxes = result[0].boxes.xyxy.cpu().numpy()
    names = result[0].names
    classes = result[0].boxes.cls.cpu().numpy().astype(int)


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
        cv2.rectangle(img, (x1,y1), (a,b), color, 2)
        cv2.putText(img, class_name, (x1, y1- 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for x, y in kps:
            idx += 1
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
            cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
