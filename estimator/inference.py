from ultralytics import YOLO
import cv2

model_path = "./estimator/weights/best.pt"

img_path = "./estimator/inferenceImages/2.png"
img = cv2.imread(img_path)

model = YOLO(model_path)

result = model(img_path)[0]
keypoints = result.keypoints.xy.cpu().numpy()
bboxes = result.boxes.xyxy.cpu().numpy()


for kps, box in zip(keypoints, bboxes):
    idx = 0
    x, y, a, b = map(int, box)
    cv2.rectangle(img, (x,y), (a,b), (0,255,0), 2)
    for x, y in kps:
        idx += 1
        cv2.circle(img, (int(x), int(y)), 4, (0,0,255), -1)
        cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
