
// ============ POSE ESTIMATION API ============
import type { APIModule } from '../apiContent'

export const poseEstimationAPIClasses: APIModule = {
  id: 'pose-estimation',
  name: 'Pose Estimation Module',
  description: 'YOLO-based detection, PnP pose solving, and error metric calculation for 6D spacecraft pose estimation.',
  classes: [
    {
      path: 'constants',
      name: 'Constants',
      file: 'estimator/solver/modules/utils/constants.py',
      description: 'Holds camera intrinsic parameters and other constants for pose estimation. The default values are based on the Intel RealSense camera',
      members: 
      [
        {
          name: 'fx',
          type: 'float',
          description: 'Camera focal length in pixels along x-axis',
          default: '915.5166015625'
        },
        {
          name: 'fy',
          type: 'float',
          description: 'Camera focal length in pixels along y-axis',
          default: '915.607421875'
        },
        {
          name: 'cx',
          type: 'float',
          description: 'Principal point x-coordinate in pixels',
          default: '629.287109375'
        },
        {
          name: 'cy',
          type: 'float',
          description: 'Principal point y-coordinate in pixels',
          default: '356.802307128906'
        },
        {
          name: 'cam_mat',
          type: 'np.ndarray',
          description: 'Camera intrinsic matrix (3x3) used for PnP solving',
          default: `np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float32)`,
        },
        {
          name: 'dist_coeffs',
          type: 'np.ndarray',
          description: 'Camera distortion coefficients (k1, k2, p1, p2, k3)',
          default: `np.zeros((5, 1), dtype=np.float32)  # Assuming no lens distortion`,
        }
      ]
    },
    {
      path: 'yolo-detect',
      name: 'YOLODetect',
      file: 'estimator/solver/modules/utils/yolo.py',
      description: 'Wrapper around Ultralytics YOLO model for object detection and keypoint extraction with confidence filtering. Designed to extract classes predicted, keypoints, keypoints confidence, bounding boxes and bounding boxes confidence from input images.',
      methods: [
        {
          name: '__init__',
          description: 'Load YOLO model from file path.',
          signature: '__init__(self, model_path: str)',
          parameters: [
            {
              name: 'model_path',
              type: 'str',
              description: 'Path to YOLO weights file (.pt format)',
            },
          ],
          returns: {
            type: 'None',
            description: 'Initializes YOLO model and loads weights',
          },
          example: `yolo = YOLODetect('./estimator/weights/format_3.4/best.pt')`,
          raises: [
            {
              type: 'FileNotFoundError',
              description: 'If model_path does not exist',
            },
          ],
        },
        {
          name: 'get_class_names',
          description: 'Retrieve trained model class labels mapping.',
          signature: 'get_class_names(self) -> Dict[int, str]',
          parameters: [],
          returns: {
            type: 'Dict[int, str]',
            description: 'Mapping of class ID integers to class name strings',
          },
          example: `names = yolo.get_class_names()
# Output: {0: 'faceA', 1: 'faceB', ...}`,
        },
        {
          name: 'get_model',
          description: 'Get reference to underlying YOLO model object.',
          signature: 'get_model(self) -> YOLO',
          parameters: [],
          returns: {
            type: 'YOLO',
            description: 'Ultralytics YOLO model instance',
          },
          example: `model = yolo.get_model()
# Access low-level YOLO API`,
        },
        {
          name: 'run_inference',
          description: 'Execute YOLO model inference on image to extract detections, keypoints, and bounding boxes.',
          signature: 'run_inference(self, image: np.ndarray, conf: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]',
          parameters: [
            {
              name: 'image',
              type: 'np.ndarray',
              description: 'Input image as BGR numpy array with shape (H, W, 3) and dtype uint8',
            },
            {
              name: 'conf',
              type: 'float',
              description: 'Confidence threshold for filtering detections',
              optional: true,
              default: '0.6',
            },
          ],
          returns: {
            type: 'Tuple[classes, keypoints, kpts_conf, bboxes, bboxes_conf]',
            description: 'Five numpy arrays: class IDs, keypoint coords, keypoint confidences, bounding boxes (xyxy), box confidences',
          },
          example: `frame = cv2.imread('frame.png')
classes, kpts, kpts_conf, bboxes, boxes_conf = yolo.run_inference(frame, conf=0.5)

# classes: shape (N,) - class IDs for each detection
# kpts: shape (N, K, 2) - x,y coords for K keypoints per detection  
# kpts_conf: shape (N, K) - confidence per keypoint
# bboxes: shape (N, 4) - [x_min, y_min, x_max, y_max]
# boxes_conf: shape (N,) - confidence per detection box`,
          notes: [
            'Keypoints in image pixel coordinates',
            'Bounding boxes in (x_min, y_min, x_max, y_max) format (not normalized)',
            'All outputs are CPU numpy arrays',
            'Confidence threshold applies to bounding boxes, not keypoints',
            'Image should be BGR (OpenCV format), not RGB',
          ],
        },
      ],
    },
    {
      path: 'pose-solver',
      name: 'PoseSolver',
      file: 'estimator/solver/modules/utils/pnp.py',
      description: 'Perspective-n-Point (PnP) solver for 6D pose estimation from 2D-3D keypoint correspondences. Supports single and multi-class objects. This is a persistent class that holds loaded 3D keypoint data and class names for use across multiple pose solving calls. The solvepose method includes adaptive initialization and coordinate frame correction to improve robustness.',
      members: [
        {
          name: 'keypoints_3d',
          type: 'Dict[str, List[Dict[str, Any]]]',
          description: 'Class variable holding loaded 3D keypoint coordinates for each object class. Structured as a dictionary mapping class names to lists of keypoint dicts with "name" and "location" fields.',
        },
        {
          name: 'class_names',
          type: 'Dict[int, str]',
          description: 'Class variable mapping YOLO class ID integers to class name strings. This allows the solver to match detected classes to their corresponding 3D keypoints for pose estimation.',
        },
      ],
      methods: [
        {
          name: 'initialize',
          description: 'Load 3D keypoint coordinates and class names (class method). Only needs to be called once to set up class variables for keypoints and class names. This allows the solvepose method to access the 3D model data without needing to reload it each time.',
          signature: '@classmethod initialize(cls, coordsfile: str, class_names: Dict[int, str]) -> None',
          parameters: [
            {
              name: 'coordsfile',
              type: 'str',
              description: 'Path to JSON file containing 3D keypoint coordinates',
            },
            {
              name: 'class_names',
              type: 'Dict[int, str]',
              description: 'Dictionary mapping class IDs to class name strings from YOLO model',
            },
          ],
          returns: {
            type: 'None',
            description: 'Loads keypoint data and class names into class variables: keypoints_3d and class_names',
          },
          example: `coords_file = './estimator/model/coords/format_3/coords.json'
class_names = {0: 'Body', 1: 'Solar_Panel'}
PoseSolver.initialize(coords_file, class_names)

# Example JSON format of coords.json:
{
    "faceA": [
        {
            "name": "kpt_a_bottom_left",
            "location": [
                0.03752017021179199,
                0.051833610981702805,
                0.008623361587524414
            ]
        },
        ...
      ],
    "faceB": [
      ...
    ],
    ...
}
`,
          notes: [
            'Must be called before any solve methods',
            'Sets class variables: keypoints_3d, class_names',
            'JSON file format: List of {name, location} objects',
          ],
        },
        {
          name: 'solvepose',
          description: 'Low-level PnP solver with adaptive initialization and coordinate frame correction.',
          signature: '@classmethod solvepose(cls, object_points, image_points, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front=True, isCoPlanar=False) -> Tuple[bool, np.ndarray | None, np.ndarray | None]',
          parameters: [
            {
              name: 'object_points',
              type: 'np.ndarray',
              description: '3D points in object frame, shape (N, 3) float32, N >= 4',
            },
            {
              name: 'image_points',
              type: 'np.ndarray',
              description: '2D points in image frame, shape (N, 2) float32, must match object_points count',
            },
            {
              name: 'rvec',
              type: 'np.ndarray',
              description: 'Initial rotation vector (3, 1) for warm-starting',
              optional: true,
            },
            {
              name: 'tvec',
              type: 'np.ndarray',
              description: 'Initial translation vector (3, 1) for warm-starting',
              optional: true,
            },
            {
              name: 'use_Extrinsic_Guess',
              type: 'bool',
              description: 'If True, use provided rvec/tvec as initialization',
              optional: true,
              default: 'False',
            },
            {
              name: 'bring_object_to_front',
              type: 'bool',
              description: 'If True, enforce positive Z by mirroring if needed',
              optional: true,
              default: 'True',
            },
            {
              name: 'isCoPlanar',
              type: 'bool',
              description: 'If True, use SOLVEPNP_IPPE (co-planar points), else ITERATIVE',
              optional: true,
              default: 'False',
            },
          ],
          returns: {
            type: 'Tuple[success, rvec, tvec]',
            description: 'Boolean success flag, rotation vector (3,1), translation vector (3,1). Returns (False, None, None) on failure.',
          },
          example: `success, rvec, tvec = PoseSolver.solvepose(
    object_points_3d,
    image_points_2d,
    isCoPlanar=False,
    bring_object_to_front=True
)

if success:
    print(f"Translation: {tvec.T} meters")
    print(f"Rotation (Rodrigues): {rvec.T}")`,
          notes: [
            'Requires camera matrix and distortion coeffs in Constants.cam_mat',
            'Automatic coordinate frame correction if Z < 0',
            'Includes timing instrumentation (CPU and wall-clock printed to stdout)',
            'Minimum 4 points required for solvability',
            'Uses OpenCV SOLVEPNP algorithm internally',
          ],
        },
        {
          name: 'format_multi_class_keypoints',
          description: 'Format multi-class keypoints from YOLO output with confidence filtering.',
          signature: '@classmethod format_multi_class_keypoints(cls, keypoints_predicted, classes_predicted, kpts_conf=None) -> Tuple[np.ndarray, np.ndarray]',
          parameters: [
            {
              name: 'keypoints_predicted',
              type: 'np.ndarray',
              description: 'YOLO keypoints shape (N_detections, N_keypoints, 2) in pixel coords',
            },
            {
              name: 'classes_predicted',
              type: 'np.ndarray',
              description: 'Class IDs from YOLO shape (N_detections,)',
            },
            {
              name: 'kpts_conf',
              type: 'np.ndarray',
              description: 'Per-keypoint confidences shape (N_detections, N_keypoints)',
              optional: true,
            },
          ],
          returns: {
            type: 'Tuple[object_points, image_points]',
            description: 'Filtered 3D and 2D points as float32 numpy arrays',
          },
          example: `obj_pts, img_pts = PoseSolver.format_multi_class_keypoints(
    kpts, classes, kpts_conf=kpts_conf
)`,
          notes: [
            'Filters keypoints with confidence < 0.5',
            'Matches YOLO keypoints to 3D model keypoints by class',
            'Returns matched point pairs only',
          ],
        },
        {
          name: 'format_multi_class_keypoints_and_solve_pose',
          description: 'Format multi-class keypoints and solve pose in one call.',
          signature: '@classmethod format_multi_class_keypoints_and_solve_pose(cls, keypoints_predicted, classes_predicted, kpts_conf=None, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front=True) -> Tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]',
          parameters: [
            {
              name: 'keypoints_predicted',
              type: 'np.ndarray',
              description: 'YOLO keypoints',
            },
            {
              name: 'classes_predicted',
              type: 'np.ndarray',
              description: 'YOLO class IDs',
            },
            {
              name: 'kpts_conf',
              type: 'np.ndarray',
              description: 'Keypoint confidences',
              optional: true,
            },
            {
              name: 'rvec',
              type: 'np.ndarray',
              description: 'Initial rotation estimate',
              optional: true,
            },
            {
              name: 'tvec',
              type: 'np.ndarray',
              description: 'Initial translation estimate',
              optional: true,
            },
            {
              name: 'use_Extrinsic_Guess',
              type: 'bool',
              description: 'Enable warm-starting',
              optional: true,
              default: 'False',
            },
            {
              name: 'bring_object_to_front',
              type: 'bool',
              description: 'Enforce positive Z',
              optional: true,
              default: 'True',
            },
          ],
          returns: {
            type: 'Tuple[success, rvec, tvec, obj_points, img_points]',
            description: 'Success flag, rotation vector, translation vector, 3D points used, 2D points used',
          },
          example: `success, rvec, tvec, obj_pts, img_pts = PoseSolver.format_multi_class_keypoints_and_solve_pose(
    kpts, classes, kpts_conf=kpts_conf
)
if success:
    print(f"Pose solved with {len(obj_pts)} keypoint correspondences")`,
          notes: [
            'Combines formatting and solving in single call',
            'Uses SOLVEPNP_IPPE for single-class, ITERATIVE for multi-class',
            'Returns filtered point sets used for solving',
          ],
        },
        {
          name: 'format_single_class_keypoints',
          description: 'Format single-class keypoints by matching all model keypoints.',
          signature: '@classmethod format_single_class_keypoints(cls, keypoints_predicted) -> Tuple[np.ndarray, np.ndarray]',
          parameters: [
            {
              name: 'keypoints_predicted',
              type: 'np.ndarray',
              description: 'YOLO keypoints shape (N_detections, N_keypoints, 2)',
            },
          ],
          returns: {
            type: 'Tuple[object_points, image_points]',
            description: 'All 3D model keypoints and corresponding 2D predictions',
          },
          example: `obj_pts, img_pts = PoseSolver.format_single_class_keypoints(kpts)`,
        },
        {
          name: 'format_single_class_keypoints_and_solve_pose',
          description: 'Format single-class keypoints and solve pose.',
          signature: '@classmethod format_single_class_keypoints_and_solve_pose(cls, keypoints_predicted, rvec=None, tvec=None, use_Extrinsic_Guess=False, bring_object_to_front=True) -> Tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]',
          parameters: [
            {
              name: 'keypoints_predicted',
              type: 'np.ndarray',
              description: 'YOLO keypoints',
            },
            {
              name: 'rvec',
              type: 'np.ndarray',
              description: 'Initial rotation',
              optional: true,
            },
            {
              name: 'tvec',
              type: 'np.ndarray',
              description: 'Initial translation',
              optional: true,
            },
            {
              name: 'use_Extrinsic_Guess',
              type: 'bool',
              description: 'Enable warm-start',
              optional: true,
            },
            {
              name: 'bring_object_to_front',
              type: 'bool',
              description: 'Enforce positive Z',
              optional: true,
            },
          ],
          returns: {
            type: 'Tuple[success, rvec, tvec, obj_points, img_points]',
            description: 'Pose solution with point correspondences',
          },
          example: `success, rvec, tvec, obj_pts, img_pts = PoseSolver.format_single_class_keypoints_and_solve_pose(kpts)`,
        },
      ],
    },
    {
      path: 'analyzer',
      name: 'Analyzer',
      file: 'estimator/solver/modules/utils/error_analyzer.py',
      description: 'Computes rotational and translational error metrics for pose accuracy evaluation.',
      methods: [
        {
          name: 'getRotationError',
          description: 'Compute angular difference between two rotation matrices in degrees.',
          signature: 'getRotationError(self, R1: np.ndarray, R2: np.ndarray) -> float',
          parameters: [
            {
              name: 'R1',
              type: 'np.ndarray',
              description: 'First rotation matrix with shape (3, 3), Predicted Rotation Matrix',
            },
            {
              name: 'R2',
              type: 'np.ndarray',
              description: 'Second rotation matrix with shape (3, 3), Ground Truth Rotation Matrix',
            },
          ],
          returns: {
            type: 'float',
            description: 'Angular error in degrees',
          },
          example: `analyzer = Analyzer()
R_pred = cv2.Rodrigues(rvec)[0]
R_ground_truth = cv2.Rodrigues(rvec_gt)[0]
rot_error = analyzer.getRotationError(R_pred, R_ground_truth)
print(f"Rotational Error: {rot_error:.2f}°")`,
          notes: [
            'Uses angle from axis-angle form: angle = arccos((trace - 1) / 2)',
            'Numerically stable with clipping to [-1, 1]',
            'Symmetric: error(R1, R2) == error(R2, R1)',
            'Returns NaN if input matrices are not valid rotations',
          ],
        },
        {
          name: 'getTranslationError',
          description: 'Compute Euclidean distance between two translation vectors.',
          signature: 'getTranslationError(self, t1: np.ndarray, t2: np.ndarray) -> float',
          parameters: [
            {
              name: 't1',
              type: 'np.ndarray',
              description: 'First translation vector, shape (3,) or (3,1). Predicted Translation Vector',
            },
            {
              name: 't2',
              type: 'np.ndarray',
              description: 'Second translation vector, shape (3,) or (3,1). Ground Truth Translation Vector',
            },
          ],
          returns: {
            type: 'float',
            description: 'Euclidean distance in units matching input (typically meters)',
          },
          example: `trans_error = analyzer.getTranslationError(tvec_pred, tvec_ground_truth)
print(f"Translation Error: {trans_error:.4f}m ({trans_error*1000:.2f}mm)")`,
          notes: [
            'Computes L2 norm: ||t1 - t2||',
            'Always non-negative',
            'Works with both (3,) and (3,1) array shapes',
            'Symmetric: error(t1, t2) == error(t2, t1)',
          ],
        },
      ],
    },
  ],
}
