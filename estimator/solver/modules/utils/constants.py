import os
import numpy as np

class Constants:
    fx = 915.5166015625
    fy = 915.607421875
    cx = 629.287109375
    cy = 356.802307128906

    cam_mat = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)