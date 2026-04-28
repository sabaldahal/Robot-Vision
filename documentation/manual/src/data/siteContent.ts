export const siteHeader = {
  kicker: 'Robot Vision / Technical Manual',
  title: 'Synthetic Data Generation + Pose Estimation Documentation',
  badges: ['Version 1.0', 'Workflow Reference'],
}

export const sidebarContent = {
  primaryPaths: ['data generator/', 'estimator/solver/'],
}

export const overviewContent = {
  title: 'Overview',
  description:
    'This manual explains the operational flow for generating synthetic spacecraft datasets in Blender and estimating 6D pose through YOLO detections and PnP-based geometric solving.',
  architectureStages: [
    'Blender Scene',
    'Annotation Export',
    'YOLO Inference',
    'PnP Solver',
    'Error Metrics',
  ],
}

export type QuickStartSection = {
  heading: string
  body: string
  code: string
  image?: string | string[]
}

export const quickStartContent: {
  title: string
  description: string
  stages: string[]
  introCode: string
  sections: QuickStartSection[]
} = {
  title: 'Quick Start',
  description:
    'Follow this workflow to generate synthetic spacecraft data, prepare it in Roboflow, train a YOLO pose model, and run the pose solver.',
  stages: [
    'Generate synthetic datasets from the Blender scripts.',
    'Export or upload the dataset into Roboflow.',
    'Preprocess, version, and download the training dataset.',
    'Train the YOLO pose model and keep the best weights.',
    'Run inference and pose solving with the trained checkpoint.',
  ],
      introCode: `
    # Full workflow shortcut

    cd "data generator"
    blender "path/to/blender_file.blend" --background --python "main.py"

    cd "../TRAIN/training"
    python train.py

    cd "../../estimator/solver"
    python inference.py
    python pose.py
    `,
  sections: [
    {
      heading: '1. Generate synthetic datasets',
      body: `Run the Blender generator from the data generator directory.

cd "data generator"
blender "path/to/blender_file.blend" --background --python "main.py"

The generator produces rendered frames, labels, and transformation metadata that can be reused for training and evaluation. If you want to inspect the export logic, see the dataset formatter in data generator/utils/dataformatter.py.`,
      code: `cd "data generator"
blender "path/to/blender_file.blend" --background --python "main.py"`,
    },
    {
      heading: '2. Prepare the dataset for Roboflow',
      body: `Open the dedicated Roboflow workflow page for the full upload and versioning flow.

Use the COCO-style export from the generator, then load the dataset into Roboflow for preprocessing, augmentation, splitting, and version creation. The Roboflow export should preserve the keypoint annotations needed for pose training.

The documentation page at /roboflow-workflow walks through the source annotations, import flow, and versioned outputs.`,
      code: `from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")`,
    },
    {
      heading: '3. Train the YOLO model',
      body: `Use the training notebook or script to download the Roboflow dataset and launch Ultralytics training.

cd "TRAIN/training"
python train.py

or open train.ipynb if you prefer an interactive run.

The model should be trained as a pose-capable YOLO checkpoint. After training, keep the best.pt artifact in the versioned weights directory so the solver can consume it consistently.

The /training-yolo-model page documents the recommended training flow and where the best checkpoint is stored.`,
      code: `from ultralytics import YOLO

model = YOLO("yolo26s-pose.pt")
results = model.train(data="path/to/data.yaml", epochs=100, imgsz=640)`,
    },
    {
      heading: '4. Run the pose solver',
      body: `After training finishes, switch to the solver directory and run inference or pose recovery.

cd "../estimator/solver"
python inference.py
python pose.py

For recorded video or RealSense streams, use pose_video_recorded.py or pose_video_realsense.py. These scripts load the trained weights, run YOLO detections, and pass the results into the PnP pose solver.

If you need metrics, run error_metrics_v3.py to compare estimated pose against the ground truth transformations.`,
      code: `cd "../estimator/solver"
python inference.py
python pose.py`,
    },
  ],
}

export type WorkflowCard = {
  title: string
  description: string
  file: string
  image?: string | string[]
  code?: string
}

export const roboflowWorkflowPageContent = {
  kicker: 'Training Pipeline',
  title: 'Roboflow Workflow',
  description:
    'Prepare the exported dataset, publish it in Roboflow, and download a versioned YOLO pose dataset for training.',
  stages: [
    'Upload the dataset to a Roboflow project.',
    'Apply preprocessing, splitting, and versioning.',
    'Download the YOLOv8 pose dataset artifact.',
  ],
  cards: [
    {
      title: 'Upload images and annotations',
      description:
        `After generating the dataset, upload the images and the COCO annotations file to Roboflow so you can apply preprocessing and versioning. Create a new 
        project in your Roboflow workspace. Add classes named according to the keypoint classes in your dataset. Define the keypoints for each classes.
        Upload the dataset using the Roboflow web interface or the CLI, making sure to include both the images and the _annotations.coco.json file.`,
      file: 'GENERATED_DATASET/images/ (images + _annotations.coco.json)',
      code: `# Example Roboflow CLI upload (replace placeholders)
Install
$ pip install roboflow

# Authenticate
$ roboflow login

# Import
$ roboflow import -w YOUR_WORKSPACE_NAME -p YOUR_PROJECT_NAME /path/to/data
                    `,
      image: ['/Robot-Vision/images/roboflow_classes.png', '/Robot-Vision/images/roboflow_keypoints.png'],
    },
    {
      title: 'Configure preprocessing & splits',
      description:
        'In the Roboflow project, add preprocessing resize, augmentation rules, and set the train/validation/test split. These steps improve training variety and evaluation quality.',
      file: 'Roboflow project (GUI)',
      image: '/Robot-Vision/images/roboflow_preprocessing.png',

    },
    {
      title: 'Download the versioned dataset',
      description:
        'When ready, download the versioned dataset (YOLOv8) via the Roboflow GUI or the API/CLI so it can be used for training locally or in a notebook.',
      file: 'TRAIN/training/local/datasets/',
      code: `# Example: download using Python Roboflow API (replace placeholders)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")`,
      image: '/Robot-Vision/images/roboflow_download.png',
    },
  ] as WorkflowCard[],
}

export const trainingYOLOModelPageContent = {
  kicker: 'Training Pipeline',
  title: 'Training YOLO Model',
  description:
    'Train a pose-capable YOLO checkpoint from the Roboflow-exported dataset and preserve the best weights for inference.',
  stages: [
    'Select a pose-capable Ultralytics base model.',
    'Point training to the exported data.yaml file.',
    'Run epochs, image sizing, and augmentation settings.',
    'Review metrics and plots in the run directory.',
    'Copy the best checkpoint into the solver weights folder.',
  ],
  cards: [
    {
      title: 'Download Roboflow dataset',
      description:
        'Download the versioned Roboflow dataset to a local directory so you can train with it. The dataset should be in YOLOv8 pose format with a data.yaml file.',
      file: 'TRAIN/training/local/datasets/',
      code: `from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")`,
    },
    {
      title: 'Train YOLO pose model',
      description:
        'Start the training process using Ultralytics, pointing to the downloaded dataset. The training script will log metrics and save the best checkpoint.',
      file: 'TRAIN/training/train.py',
      code: `from ultralytics import YOLO

model = YOLO("yolov11s-pose.pt")
model.train(
  data="path/to/data.yaml",
  epochs=100,
  imgsz=640,
  fliplr=0.0, 
  flipud=0.0
)`,
    },
    {
      title: 'Save best weights',
      description:
        'After training completes, copy the best checkpoint from the runs directory into the versioned weights folder for inference.',
      file: 'estimator/weights/format_3.4/',
      code: `# Best checkpoint saved to: runs/detect/train/weights/best.pt
# Copy to solver weights directory:
cp runs/detect/train/weights/best.pt \
  ../../../estimator/weights/format_3.4/best.pt`,
    },
  ] as WorkflowCard[],
}

export const syntheticDataPageContent = {
  kicker: 'Parent Section',
  title: 'Synthetic Data Generation (Blender)',
  description:
    'Browse the subsections below for focused documentation on each stage of the generation pipeline.',
}

export const poseEstimationPageContent = {
  kicker: 'Parent Section',
  title: 'Pose Estimation (YOLO + PnP)',
  description:
    'Each subsection covers one solver stage in depth, from detections to geometric recovery and metric validation.',
}

export const apiReferencePageContent = {
  kicker: 'API Documentation',
  title: 'API Reference',
  description:
    'Detailed function and class documentation for all major modules, including signatures, parameters, return types, and usage examples.',
}

export type ConfigurationRow = {
  area: string
  keyInputs: string
  expectedOutput: string
}

export const configurationContent = {
  title: 'Configuration Reference',
  rows: [
    {
      area: 'Scene Randomization',
      keyInputs: 'Camera bounds, object transforms, lighting distributions',
      expectedOutput: 'Diverse image set for robust detector training',
    },
    {
      area: 'Detection Inference',
      keyInputs: 'Model weights, confidence threshold, image stream',
      expectedOutput: 'Bounding boxes and candidate keypoints',
    },
    {
      area: 'PnP Solve',
      keyInputs: '2D-3D correspondence mapping, camera intrinsics',
      expectedOutput: 'rvec and tvec for each accepted frame',
    },
  ] as ConfigurationRow[],
}

export type OutputArtifact = {
  title: string
  description: string
}

export const outputsContent = {
  title: 'Outputs and Artifacts',
  artifacts: [
    {
      title: 'Generated Dataset',
      description: 'Image frames, labels, and metadata generated from Blender scenes.',
    },
    {
      title: 'Model Weights',
      description: 'Versioned YOLO checkpoints located under the weights directory.',
    },
    {
      title: 'Evaluation Reports',
      description: 'Rotational and translational error report',
    },
  ] as OutputArtifact[],
}

export type TroubleshootingItem = {
  title: string
  description: string
}

export const troubleshootingContent = {
  title: 'Troubleshooting',
  items: [
    {
      title: 'Can\'t understand the model weights format structure',
      description:
        'format_3.x is the current format used for YOLO weights. format_A.x (major version: A in format_A.x) specifies the structure of the keypoints and classes in the dataset while the minor version (x in format_A.x) specifies the training configuration.',
    },

  ] as TroubleshootingItem[],
}
