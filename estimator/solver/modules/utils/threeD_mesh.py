import numpy as np

def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.split()
                # Convert x, y, z coordinates to floats
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def load_obj_faces(filepath):
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()
                face_indices = []
                for p in parts[1:]:
                    # Handle cases like "1", "1/2", or "1/2/3"
                    vertex_index = int(p.split('/')[0]) - 1  # Subtract 1 for 0-based indexing
                    face_indices.append(vertex_index)
                faces.append(face_indices)
    return np.array(faces, dtype=int)

def project_points_numpy(objectPoints, rvec, tvec, cam_mat):
    """
    Project 3D points to 2D using OpenCV-style pinhole camera model.
    No distortion applied.

    Parameters:
        objectPoints : Nx3 array of 3D points
        rvec : 3x1 rotation vector (Rodrigues)
        tvec : 3x1 translation vector
        cam_mat : 3x3 camera intrinsic matrix

    Returns:
        Nx2 array of 2D points in pixel coordinates
    """
    objectPoints = np.asarray(objectPoints).reshape(-1,3)
    tvec = np.asarray(tvec).reshape(3,1)

    # Convert rvec to rotation matrix
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        R = np.eye(3)
    else:
        axis = (rvec / theta).flatten()
        x, y, z = axis
        K = np.array([[ 0, -z,  y],
                      [ z,  0, -x],
                      [-y,  x,  0]])
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

    print('R from custom project', R)
    # Transform points to camera coordinates
    points_cam = (R @ objectPoints.T) + tvec  # shape 3xN

    # Perspective division
    x = points_cam[0,:] / points_cam[2,:]
    y = points_cam[1,:] / points_cam[2,:]

    # Apply camera intrinsics
    fx, fy = cam_mat[0,0], cam_mat[1,1]
    cx, cy = cam_mat[0,2], cam_mat[1,2]

    u = fx * x + cx
    v = fy * y + cy

    points_2d = np.vstack([u,v]).T
    return points_2d

def getaxisangle(angle_axis):
    angle_axis = angle_axis.reshape(3)

    angle = np.linalg.norm(angle_axis)

    if angle > 1e-9:
        axis = angle_axis / angle
    else:
        axis = np.array([1.0, 0.0, 0.0]) 

    return axis, angle

def load_blender_matrix(filepath):
    Transformation_Matrix_Blender_to_OpenCV = np.diag([1.0, -1.0, -1.0])
    matrix_from_file = np.loadtxt(filepath)
    R_Matrix_Blender = matrix_from_file[:3, :3]
    Tvec_Blender = matrix_from_file[:3, 3].reshape(3,1)
    Rvec_Blender_to_OpenCV = Transformation_Matrix_Blender_to_OpenCV @ R_Matrix_Blender
    Tvec_Blender_to_OpenCV = (Transformation_Matrix_Blender_to_OpenCV @ Tvec_Blender).reshape(3,1)

    return Rvec_Blender_to_OpenCV, Tvec_Blender_to_OpenCV