import cv2
import numpy as np

def draw_bbox_keypoints(image, bboxes, kpts, classes, classes_name, show_image = True, wait=True):
    for kps, box, cls_id in zip(kpts, bboxes, classes):
        idx = 0
        color = list(np.random.random(size=3) * 256)
        class_name = classes_name[cls_id]
        print('class id', class_name)
        x1, y1, a, b = map(int, box)
        cv2.rectangle(image, (x1,y1), (a,b), color, 1)
        cv2.putText(image, class_name, (x1, y1- 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        for x, y in kps:
            idx += 1
            cv2.circle(image, (int(x), int(y)), 4, color, -1)
            cv2.putText(image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    if show_image:
        cv2.imshow('Bboxes and Keypoints', image)

    if wait:           
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image

def draw_confidence_scores(image, bboxes_conf, kpts_conf, classes, classes_name, show_image=True, wait=True):
    line_height = 15
    x = 10
    y = 0
    for b_conf, k_conf, cls_id in zip(bboxes_conf, kpts_conf, classes):
        color = list(np.random.random(size=3) * 256)
        class_name = classes_name[cls_id]
        cv2.putText(image, class_name, (x-5, y+line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(image, f'bbox : {b_conf:.3f}', (x, y+(line_height * 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        cv2.putText(image, f'Keypoints:', (x, y+(line_height * 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += (line_height * 3)
        for idx, kc in enumerate(k_conf):
            cv2.putText(image, f'{idx+1} : {kc:.3f}', (x+5, y+line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += line_height

        y += line_height
    if show_image:
        cv2.imshow('Confidence Scores', image)
        
    if wait:           
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image

            