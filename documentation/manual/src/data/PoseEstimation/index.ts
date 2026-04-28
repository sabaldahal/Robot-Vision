import type { DocStep } from '../docsContent'

export const estimationSteps: DocStep[] = [
  {
    title: 'Detection Inference with YOLO',
    description:
      'Run trained models to obtain spacecraft detections and confidence-filtered observations in image coordinates.',
    file: 'estimator/solver/inference.py',
    to: '/pose-estimation/detection-inference-yolo',
    notes: [
      'Consumes model assets maintained under weights/.',
      'Supports confidence thresholding for robust downstream solving.',
      'Runs predictions and visualizes bounding boxes and keypoints.',
    ],
    sections: [
      {
        heading: 'Inference Workflow',
        body: `
The inference process involves loading the trained YOLO model, and running it on input images to obtain detections.
The model outputs bounding boxes, class labels, and confidence scores for each detected object.
Post-processing steps filter detections based on confidence thresholds, and extract keypoint coordinates for use
in the subsequent pose estimation step.
\n
<b>estimator/solver/modules/utils/yolo.py</b> contains helper functions with the main logic for loading models, running inference, and 
generating outputs.
\n
<b>estimator/solver/modules/visualize/bbox_kpts_viz.py</b> provides utilities for visualizing the detection results,
including bounding boxes and keypoints, which can be used for debugging and qualitative assesment of model performance.
\n
<b>estimator/solver/inference.py</b> orchestrates the overall inference workflow, defining image path, defining model path, 
integrating model loading,
prediction execution, and result visualization into a cohesive process.
\n
The model weights are stored in <b>estimator/weights/format_3.4/best.pt</b>. Format minor version is updated with the latest trained model weights as the model is iteratively improved.
        `
      }
    ]
  },
  {
    title: 'Geometric Pose Recovery using PnP',
    description:
      'Estimate orientation and translation from 2D-3D correspondences to produce rvec and tvec for each frame.',
    file: 'estimator/solver/pose.py, estimator/solver/pose_video_recorded.py, estimator/solver/pose_video_realsense.py',
    to: '/pose-estimation/geometric-pose-recovery-pnp',
    notes: [
      'Requires consistent keypoint ordering between model and image.',
      'Produces motion-ready pose outputs for robotics and control.',
      'Can operate on single frames and longer sequences.',
    ],
    sections: [
      {
        heading: 'Pose Estimation Workflow',
        body: `
The pose estimation process uses the Perspective-n-Point (PnP) algorithm to compute the 6-DoF pose of the spacecraft from 2D-3D correspondences.
The 3D model points are defined in the spacecraft coordinate frame, while the 2D image points are obtained from the YOLO detection step. The 3D model points
are defined in <b>estimator/model/coords/format_3/coords.json</b>.
The PnP algorithm estimates the rotation vector (rvec) and translation vector (tvec) that describe the pose of the spacecraft relative to the camera.
\n
<b>estimator/solver/modules/utils/pnp.py</b> contains the core logic for performing PnP pose estimation on individual frames. It handles 
the formatting of the predicted keypoints and actual 3D model points into a format suitable for OpenCV's solvePnP function. This mainly involves
matching the predicted keypoints to the corresponding 3D model points based on their defined ordering, filtering out low-confidence predictions,
and filtering out keypoints of the class that is not being predicted. For example, if the model predicts classes faceA and faceB, all the keypoints
belonging to rest of the classes will be filtered out and the remaining keypoints will be matched to the corresponding 3D model points.
\n
After the data is formatted, solvePnP is called to compute the pose using the image_points(predicted keypoints) and object_points(actual 3D keypoints).
<b>PoseSolve.solvepose</b> is the main function that implements this workflow. Initial rvec and tvec (use_Extrinsic_Guess must be set to True)can be provided for better initialization,
which can improve accuracy but is not guaranteed. This script also handles pose calculation for co-planar objects for which it uses the SOLVEPNP_IPPE method, and for non co-planar
objects it uses the SOLVEPNP_ITERATIVE method. This is handled automatically when the pose is calculated by invoking the
function: <b>PoseSolve.format_multi_class_keypoints_and_solve_pose</b>. Additionally, this function also handles filtering out the 
keypoints with low confidence scores. Finally, <b>PoseSolve.solvepose</b> also deals with the case when the solved pose is a mirror
solution and recalculates the pose again by feeding the initial solution of rvec and tvec by slightly modifying them (creating a nearly mirrored rvec and tvec).
\n
<b>estimator/solver/pose.py</b> applies the PnP pose estimation to an individual frame. It loads the utility functions
from <b>estimator/solver/modules/utils/pnp.py</b> and defines the main workflow for running pose estimation on a single image, including loading the image, running
inference to get keypoints,,
and then calling the PnP utility functions to compute the pose.
\n
The output of the pose estimation step is the rvec and tvec for each frame.
<b>rvec</b> is a 3D vector in axis-angle representation that describes the rotation of the spacecraft,
while <b>tvec</b> is a 3D vector that describes the translation of the spacecraft relative to the camera.
        `
      },
      {
        heading: 'Pose Estimation on Video Sequences',
        body: `
The pose estimation process can also be applied to video sequences. This is implemented in
<b>estimator/solver/pose_video_recorded.py</b> for recorded videos and <b>estimator/solver/pose_video_realsense.py</b> for live video streams from a RealSense camera.
The workflow is similar to the single frame case, but it iterates over each frame of the video, running inference and pose estimation sequentially.
This allows for continuous tracking of the spacecraft's pose over time.
        `
      },
      {
        heading: 'Pose Estimation Visualization',
        body: `
Visualization of the estimated pose can be done by projecting the 3D model points back onto the image using the computed rvec and tvec.
<b>estimator/solver/modules/visualize/pose_viz.py</b> provides utilities for visualizing the estimated pose, including drawing the coordinate axes and the 3D model overlay on the image.
The helper function defined in this script is used in all the pose estimation scripts <b>estimator/solver/pose.py</b>, <b>estimator/solver/pose_video_recorded.py</b>, 
and <b>estimator/solver/pose_video_realsense.py</b> to visualize the estimated pose on the output images and videos.
        `
      }
    ]
  },
  {
    title: 'Evaluation and Error Metrics',
    description:
      'Measure rotational and translational error across test dataset.',
    file: 'estimator/solver/error_metrics_v3.py',
    to: '/pose-estimation/evaluation-error-metrics',
    notes: [
      'Calculates rotational error in degrees and translational error in meters.',
      'Provides objective benchmarks for model promotion decisions.',
      'Supports confidence diagnostics.',
    ],
    sections: [
      {
        heading: 'Error Metrics Calculation',
        body: `
The evaluation of pose estimation performance is done by calculating error metrics that quantify the accuracy of the estimated pose compared to the ground truth.
<b>estimator/solver/error_metrics_v3.py</b> contains the logic for calculating these error metrics, including both rotational and translational errors.
The rotational error is calculated in degrees by comparing the estimated rotation (rvec) with the ground truth rotation, while the translational error is calculated in meters by comparing the estimated translation (tvec) with the ground truth translation.
These metrics provide objective benchmarks for assessing model performance and making informed decisions about model promotion.
\n
The script first runs inference, then runs pose estimation to get the rvec and tvec for each frame.
\n
Ground truth data is obtained from the synthetic dataset. The folder <b>transformation_matrices</b> obtained from the synthetic dataset generation step contains
the ground truth transfomation matrices with respect to the camera frame for each image in the test dataset. Rotational matrix and translation vector are extracted
from these matrices and used as ground truth for calculating the error metrics. rvec and tvec obtained from the pose estimation step are compared against these ground truth.
rvec is converted to a rotation matrix before calculating the error metrics.
        `
      }
    ]
  },
]