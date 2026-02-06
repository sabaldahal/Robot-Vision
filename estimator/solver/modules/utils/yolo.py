from ultralytics import YOLO


class YOLODetect:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_class_names(self):
        return self.model.names
    
    def get_model(self):
        return self.model
    
    def run_inference(self, image, conf=0.6):
        result = self.model(image, conf=conf)[0]
        keypoints = result.keypoints.xy.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()
        classes_predicted = result.boxes.cls.cpu().numpy().astype(int)
        boxes_conf = result.boxes.conf.cpu().numpy()
        kps_conf = result.keypoints.conf.cpu().numpy()

        return classes_predicted, keypoints, kps_conf, bboxes, boxes_conf