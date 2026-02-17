import cv2
import numpy as np
from modules.utils import constants

def draw_pose(frame, rvec, tvec, vertices_array, faces_array, wait=True, window = "Pose Visualization"):
    
    axis_length = 0.05  # meters
    axis_points = np.float32([
        [0, 0, 0],                   # origin
        [axis_length, 0, 0],         # X axis (red)
        [0, axis_length, 0],         # Y axis (green)
        [0, 0, axis_length]          # Z axis (blue)
    ])


    # Project to image
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, constants.Constants.cam_mat, constants.Constants.dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    objtoimg, _ = cv2.projectPoints(vertices_array, rvec, tvec, constants.Constants.cam_mat, constants.Constants.dist_coeffs)
    objtoimg = np.int32(objtoimg).reshape(-1, 2)

    #Draw faces
    for face in faces_array:
        pts = objtoimg[face]
        cv2.polylines(frame, [pts], True, (0,255,255), 2)

    # Draw axes on frame
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3) # X - red
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3) # Y - green
    cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3) # Z - blue
    cv2.imshow(window, frame)

    if wait:            
        cv2.waitKey(0)
        cv2.destroyAllWindows()